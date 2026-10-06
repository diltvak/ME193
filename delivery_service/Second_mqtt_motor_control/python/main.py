from arduino.app_utils import App, Bridge
from mqttlib import MQTTClient
import json
import math
import time

TOPIC = "position"

# --- Controller tuning ---------------------------------------------------
KP_X = 0.5           # drive speed (0..1) per unit of X error
KP_DIST = 0.0        # (disabled) 2D distance gain, used only by compute_full
KP_TURN = 0.0        # (disabled) turn gain, used only by compute_full
MAX_SPEED = 0.6      # cap on drive speed (0..1)
MIN_FORWARD = 0.0    # (disabled) heading-discovery creep, used only by compute_full
MIN_PWM = 60         # smallest PWM (0..255) that actually moves the motors
KD_X = 0.3           # damping: drive speed (0..1) per unit/s of X velocity
D_MAX = 5.0         # cap on the derivative contribution (0..1)
D_FILTER = 0.25       # low-pass on the velocity estimate (0 = frozen, 1 = raw)
DEADBAND = 0.05      # stop when |x| is smaller than this
MOVE_EPS = 0.02      # min displacement between messages to trust it for heading
STALE_S = 1.0        # stop if no message arrives for this long

# Both signs flip a motor's direction if it is wired the "wrong" way round.
LEFT_SIGN = 1        # motor on pins 5 & 6
RIGHT_SIGN = 1       # motor on pins 10 & 11

latest = None        # (x, y, theta_or_None, timestamp), set on the MQTT thread
heading = None       # best estimate of robot heading in radians (0 = +x axis)
prev_xy = None
d_prev = None        # (x, timestamp) of the last message used for the derivative
d_vel = 0.0          # filtered x velocity (units/s)
shown_x = None       # last x value sent to the LED matrix
shown_ring = None    # last ring state sent to the LED matrix


def on_position(topic, payload):
    """Parse {"x":..,"y":..,["theta":..]} and update the heading estimate."""
    global latest, heading, prev_xy
    try:
        data = json.loads(payload)
        x, y = float(data["x"]), float(data["y"])
        theta = float(data["theta"]) if "theta" in data else None
    except (ValueError, KeyError, TypeError):
        print(f"Ignoring bad message on [{topic}]: {payload}")
        return

    if theta is not None:
        heading = theta
    elif prev_xy is not None:
        dx, dy = x - prev_xy[0], y - prev_xy[1]
        if math.hypot(dx, dy) > MOVE_EPS:
            heading = math.atan2(dy, dx)
    if theta is not None or prev_xy is None or math.hypot(x - prev_xy[0], y - prev_xy[1]) > MOVE_EPS:
        prev_xy = (x, y)
    latest = (x, y, theta, time.time())


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def to_pwm(u):
    """Map a -1..1 command to a signed PWM value with a motor dead-zone offset."""
    u = max(-1.0, min(1.0, u))
    if abs(u) < 0.01:
        return 0
    pwm = MIN_PWM + abs(u) * (255 - MIN_PWM)
    return int(math.copysign(pwm, u))


def compute(x, y, t):
    """X-only controller: drive straight forward/back to bring x to 0.

    No turning and no Y motion; y is ignored. Positive x -> reverse, negative
    x -> forward (flip KP_X's sign if the robot's axis is mirrored).
    A clamped, filtered derivative term brakes as x closes on the setpoint.
    """
    global d_prev, d_vel
    # Update velocity only on a new message with a sane dt (else hold last value)
    if d_prev is None:
        d_vel = 0.0
        d_prev = (x, t)
    elif t - d_prev[1] > 1e-3:
        dt = t - d_prev[1]
        if dt < STALE_S:
            d_vel += D_FILTER * ((x - d_prev[0]) / dt - d_vel)
        else:
            d_vel = 0.0
        d_prev = (x, t)

    if abs(x) < DEADBAND:
        return 0.0, 0.0
    p = -KP_X * x
    d = max(-D_MAX, min(D_MAX, -KD_X * d_vel))
    u = p + d
    if u * p < 0:      # never let damping reverse the drive direction
        u = 0.0
    u = max(-MAX_SPEED, min(MAX_SPEED, u))
    return u, u


def compute_full(x, y):
    """(Unused) full 2D controller with heading/turning: returns (left, right) in -1..1."""
    dist = math.hypot(x, y)
    if dist < DEADBAND:
        return 0.0, 0.0

    if heading is None:
        # Heading unknown: drive straight so the next position reveals it.
        return MIN_FORWARD, MIN_FORWARD

    desired = math.atan2(-y, -x)       # direction from robot to origin
    err = wrap(desired - heading)      # + = origin is to the left (CCW)

    v = min(MAX_SPEED, KP_DIST * dist)
    v *= max(0.0, math.cos(err))       # slow down when pointing away
    v = max(v, MIN_FORWARD)
    w = KP_TURN * err

    left, right = v - w, v + w
    m = max(abs(left), abs(right), 1.0)  # keep the ratio if one side saturates
    return left / m, right / m


def loop():
    # The MCU can still be rebooting/registering its functions right after a
    # flash, so a failed Bridge call is retried on the next tick, not fatal.
    try:
        step()
    except ValueError as e:
        print(f"Bridge call failed (MCU not ready?), retrying: {e}")
        time.sleep(0.5)


def step():
    global shown_x, shown_ring, d_prev
    snap = latest
    no_data = snap is None or time.time() - snap[3] > STALE_S

    # Outer ring of the LED matrix lights up while no MQTT data is arriving
    if no_data != shown_ring:
        Bridge.call("show_ring", no_data)
        shown_ring = no_data

    if snap is not None:
        # LED matrix shows x as an integer percentage (x * 100), clamped to +/-99
        x_disp = max(-99, min(99, round(snap[0] * 100)))
        if x_disp != shown_x:
            Bridge.call("show_number", x_disp)
            shown_x = x_disp

    if no_data:
        d_prev = None
        Bridge.call("set_motors", 0, 0)
    else:
        left, right = compute(snap[0], snap[1], snap[3])
        Bridge.call("set_motors", LEFT_SIGN * to_pwm(left), RIGHT_SIGN * to_pwm(right))
    time.sleep(0.05)


client = MQTTClient()
client.connect()
client.subscribe(TOPIC, on_position)
print(f"Listening on MQTT topic '{TOPIC}' at {client.broker}")

try:
    App.run(user_loop=loop)
finally:
    try:
        Bridge.call("set_motors", 0, 0)
    except Exception:
        pass
    client.disconnect()
