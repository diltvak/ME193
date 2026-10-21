"""Sumo robot: LEGO Controller drives the Cytron motors; a py_trees behavior tree watches
the arena boundary and, when the robot leaves the ring, stops it and publishes "Defeat" over MQTT.

Boundary logic: the front LEGO Color Sensor sees the blue tape. Seeing blue and then no longer
seeing blue is a *crossing*. Because "no blue" looks the same inside and outside the ring, the
MPU6050 x-axis (positive = forward) is used to tell what the crossing was:
  - moving forward  almost the whole time on the tape -> left the arena forwards  -> Defeat
  - moving backward almost the whole time on the tape -> left the arena backwards -> Defeat
  - otherwise (direction changed)                      -> recovered, keep fighting

Pusher: a fast "punch" of the hand-held Double Motor IMU extends the Single Motor
pusher for PUSH_S seconds, then retracts it (opposite direction) for the same time."""
import math
import time

import py_trees
from py_trees.common import Status

from arduino.app_utils import App, Bridge, Logger

from lego_ble import LegoGateway
from mqttlib import MQTTClient

logger = Logger("sumo")
DEADBAND = 5  # percent
PERIOD_S = 0.02
BLUE_ID = 3        # LEGO SDK color number for blue

MQTT_TOPIC = "ME193/Sumo"
DEFEAT_MESSAGE = "Defeat"

# --- Boundary detection (tune on the real robot) ---
G = 9.81
BLUE_LOST_S = 0.10          # blue must be absent this long to count as "stopped seeing blue"
MOVING_MPS = 0.05           # |estimated velocity| above this counts as moving forward/backward
DIRECTION_FRACTION = 0.8    # "just about the whole time" = at least this share of the time on blue
VEL_LEAK_TAU_S = 2.0        # velocity estimate decays toward 0 with this time constant (drift control)
STOPPED_HOLD_S = 0.3        # wheels commanded to 0 this long -> robot assumed at rest (zero velocity, re-bias)
BIAS_ALPHA = 0.02           # accel bias tracking rate while at rest (per tick)

PUNCH_ACC_THRESHOLD = 1500  # raw IMU units; uncalibrated, tune using the logged peaks
PUNCH_MIN_GYRO = 0
PUSH_S = 0.5               # extend time, and retract time
PUSH_SPEED = 100           # percent
PUSHER_EXTEND_DIRECTION = "clockwise"  # flip if the pusher extends the wrong way
PUSHER_RETRACT_DIRECTION = "counterclockwise" if PUSHER_EXTEND_DIRECTION == "clockwise" else "clockwise"
PUNCH_COOLDOWN_S = 0.3     # after a full extend+retract cycle

lego = LegoGateway()
mqtt = MQTTClient()
_last_log = 0.0
_last_imu_log = 0.0
# Pusher cycle: "idle" -> "extending" -> "retracting" -> "idle"
_pusher_state = "idle"
_pusher_phase_end = 0.0
_pusher_cycle_done = 0.0


def pwm(lever):
    if lever is None or abs(lever) < DEADBAND:
        return 0
    return round(max(-100, min(100, lever)) * 2.55)


def update_pusher(imu, pusher_connected, now):
    """Non-blocking punch -> extend -> retract state machine (the gateway times each move)."""
    global _pusher_state, _pusher_phase_end, _pusher_cycle_done, _last_imu_log
    if _pusher_state == "extending" and now >= _pusher_phase_end:
        lego.pusher_run_for_time(int(PUSH_S * 1000), PUSHER_RETRACT_DIRECTION, PUSH_SPEED)
        _pusher_state, _pusher_phase_end = "retracting", now + PUSH_S
    elif _pusher_state == "retracting" and now >= _pusher_phase_end:
        _pusher_state, _pusher_cycle_done = "idle", now
    if _pusher_state != "idle":
        return
    acc, gyro = imu["peak_acc"] or 0.0, imu["peak_gyro"] or 0.0
    # Calibration aid: log noticeable movement so the threshold can be tuned.
    if imu["connected"] and acc >= 0.4 * PUNCH_ACC_THRESHOLD and now - _last_imu_log > 0.2:
        _last_imu_log = now
        logger.info(f"IMU peak acc={acc:.0f} gyro={gyro:.0f} (threshold acc={PUNCH_ACC_THRESHOLD})")
    if (
        imu["connected"] and pusher_connected
        and now - _pusher_cycle_done >= PUNCH_COOLDOWN_S
        and acc >= PUNCH_ACC_THRESHOLD and gyro >= PUNCH_MIN_GYRO
    ):
        logger.info(f"PUNCH detected (acc={acc:.0f}) -> pusher extend")
        lego.pusher_run_for_time(int(PUSH_S * 1000), PUSHER_EXTEND_DIRECTION, PUSH_SPEED)
        _pusher_state, _pusher_phase_end = "extending", now + PUSH_S


class Ctx:
    """State shared by every behavior in the tree (one tick's worth of sensor data + arena tracking)."""

    def __init__(self):
        self.snap = None
        self.now = 0.0
        self.dt = PERIOD_S
        self.ax = None             # latest accel x in g (None if the IMU did not answer)
        self.bias = None           # accel x at rest (gravity component + offset), g
        self.velocity = 0.0        # estimated forward velocity, m/s (+ = forward)
        self.wheels = (0, 0)       # last commanded wheel PWM
        self.stopped_since = None  # when the wheels were last commanded to 0
        # Blue-crossing tracker
        self.on_blue = False
        self.blue_seen_at = 0.0
        self.blue_lost_since = None
        self.fwd_t = self.rev_t = self.total_t = 0.0
        self.last_crossing = None  # "recovered" / "forward" / "backward" (informational)
        # Result
        self.defeat_direction = None  # None while still in the ring; "forward"/"backward" once out
        self.defeat_sent = False


ctx = Ctx()


def set_wheels(left, right):
    Bridge.notify("drive", left, right)
    ctx.wheels = (left, right)
    if left or right:
        ctx.stopped_since = None
    elif ctx.stopped_since is None:
        ctx.stopped_since = ctx.now


# ---------------------------------------------------------------- behaviors

class ReadSensors(py_trees.behaviour.Behaviour):
    """Snapshot LEGO devices and read the MPU6050 x acceleration."""

    def update(self):
        ctx.now = time.monotonic()
        ctx.snap = lego.snapshot()
        try:
            ax = Bridge.call("get_accel_x")
            ctx.ax = None if ax != ax else ax  # NaN: MPU6050 not responding
        except Exception as e:
            logger.warning(f"get_accel_x failed: {e}")
            ctx.ax = None
        return Status.SUCCESS


class EstimateVelocity(py_trees.behaviour.Behaviour):
    """Leaky integration of bias-corrected x acceleration -> forward velocity (+ = forward)."""

    def __init__(self, name):
        super().__init__(name)
        self._prev = None

    def update(self):
        dt = ctx.now - self._prev if self._prev is not None else PERIOD_S
        self._prev = ctx.now
        ctx.dt = dt = max(0.001, min(dt, 0.2))
        if ctx.ax is None:
            return Status.SUCCESS
        if ctx.bias is None:
            ctx.bias = ctx.ax
        at_rest = ctx.stopped_since is not None and ctx.now - ctx.stopped_since >= STOPPED_HOLD_S
        if at_rest:
            ctx.velocity = 0.0
            ctx.bias += BIAS_ALPHA * (ctx.ax - ctx.bias)
        else:
            ctx.velocity += (ctx.ax - ctx.bias) * G * dt
            ctx.velocity *= math.exp(-dt / VEL_LEAK_TAU_S)
        return Status.SUCCESS


class TrackBlueCrossing(py_trees.behaviour.Behaviour):
    """Watch for "blue, then no blue" and classify it with the IMU direction history."""

    def update(self):
        color = ctx.snap["color"]
        blue = bool(color["connected"] and color["color"] == BLUE_ID)
        now = ctx.now
        if blue:
            ctx.blue_lost_since = None
            if not ctx.on_blue:
                ctx.on_blue, ctx.blue_seen_at = True, now
                ctx.fwd_t = ctx.rev_t = ctx.total_t = 0.0
                logger.info("Blue tape seen")
        if ctx.on_blue:
            # Accumulate how long we moved each way while on the tape (+x = forward).
            ctx.total_t += ctx.dt
            if ctx.velocity > MOVING_MPS:
                ctx.fwd_t += ctx.dt
            elif ctx.velocity < -MOVING_MPS:
                ctx.rev_t += ctx.dt
            if not blue:
                if ctx.blue_lost_since is None:
                    ctx.blue_lost_since = now
                if now - ctx.blue_lost_since >= BLUE_LOST_S:
                    self._crossing_finished()
        return Status.SUCCESS

    def _crossing_finished(self):
        ctx.on_blue = False
        total = max(ctx.total_t, 1e-6)
        fwd, rev = ctx.fwd_t / total, ctx.rev_t / total
        summary = f"on blue {ctx.total_t:.2f}s, forward {fwd:.0%}, backward {rev:.0%}"
        if fwd >= DIRECTION_FRACTION:
            ctx.last_crossing = ctx.defeat_direction = "forward"
            logger.info(f"Left the arena moving FORWARD ({summary})")
        elif rev >= DIRECTION_FRACTION:
            ctx.last_crossing = ctx.defeat_direction = "backward"
            logger.info(f"Left the arena moving BACKWARD ({summary})")
        else:
            ctx.last_crossing = "recovered"
            logger.info(f"Direction changed on the tape -> recovered, still in the ring ({summary})")


class IsDefeated(py_trees.behaviour.Behaviour):
    def update(self):
        return Status.SUCCESS if ctx.defeat_direction is not None else Status.FAILURE


class StopWheels(py_trees.behaviour.Behaviour):
    def update(self):
        set_wheels(0, 0)
        lego.pusher_stop()
        return Status.SUCCESS


class SendDefeat(py_trees.behaviour.Behaviour):
    """Publish "Defeat" once (QoS 1 so the broker acknowledges it)."""

    def update(self):
        if not ctx.defeat_sent:
            ctx.defeat_sent = True
            logger.info(f"DEFEAT (left {ctx.defeat_direction}) -> MQTT {MQTT_TOPIC}: {DEFEAT_MESSAGE}")
            mqtt.publish(MQTT_TOPIC, DEFEAT_MESSAGE, qos=1)
        return Status.SUCCESS


class JoystickDrive(py_trees.behaviour.Behaviour):
    """Normal play: levers drive the wheels, hand-held IMU punches fire the pusher."""

    def update(self):
        global _last_log
        snap, now = ctx.snap, ctx.now
        c, color = snap["controller"], snap["color"]
        # Wheels are wired opposite to the levers, so the lever values are swapped here.
        left, right = (pwm(c["right"]), pwm(c["left"])) if c["connected"] else (0, 0)
        set_wheels(left, right)
        update_pusher(snap["imu"], snap["pusher"]["connected"], now)
        interval = 0.2 if (left or right) else 1.0
        if now - _last_log > interval:
            _last_log = now
            logger.info(
                f"gateway={lego.gateway_connected} controller={c['connected']} color={color['color']} "
                f"on_blue={ctx.on_blue} v={ctx.velocity:+.2f}m/s L={c['left']} R={c['right']} -> drive({left}, {right})"
            )
        return Status.SUCCESS


def build_tree():
    perceive = py_trees.composites.Sequence("Perceive", memory=False, children=[
        ReadSensors("Read Sensors"),
        EstimateVelocity("Estimate Velocity (IMU x)"),
        TrackBlueCrossing("Track Blue Crossing"),
    ])
    defeat = py_trees.composites.Sequence("Defeat Protocol", memory=False, children=[
        IsDefeated("Left Arena?"),
        StopWheels("Stop Wheels"),
        SendDefeat("Send 'Defeat' (MQTT)"),
    ])
    decide = py_trees.composites.Selector("Decide", memory=False, children=[
        defeat,
        JoystickDrive("Joystick Drive + Pusher"),
    ])
    return py_trees.trees.BehaviourTree(
        py_trees.composites.Sequence("Sumo Round", memory=False, children=[perceive, decide])
    )


tree = build_tree()
tree.setup(timeout=5)
logger.info("Behavior tree:\n" + py_trees.display.unicode_tree(tree.root))


def loop():
    start = time.monotonic()
    tree.tick()
    time.sleep(max(0.0, PERIOD_S - (time.monotonic() - start)))


mqtt.connect()
try:
    App.run(user_loop=loop)
finally:
    Bridge.notify("drive", 0, 0)  # stop wheels on shutdown
    lego.pusher_stop()
    mqtt.disconnect()
