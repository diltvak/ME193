"""Walk the real LEGO Double Motor robot straight and very slowly for 50 seconds, using the policy that
walk_quarter_long_rl.py trained in simulation (0.015 m/s, so about 75 cm in 50 seconds).

The same as walk_quarter_lego.py, with these changes:
  - It loads walk_quarter_long_policy.pt. That policy practised whole 50 second walks; the one in
    walk_quarter_lego.py only ever practised 10 seconds, and veered off course from about there on.
  - RUN_SECONDS is 50 instead of 100, the length of walk the policy practised (and of the simulation's film).
Everything else is unchanged, because walk_quarter_long_rl.py gives the policy the same 20 numbers and the same
motor limit as walk_quarter_rl.py: the output is cut off at -1 and +1, 1 asks an arm for MAX_MOTOR_SPEED = 2 rad/s,
drive() carries the rounded-off part of each speed over to the next step, and the arm speed is measured over 0.2 s.

The laptop runs the policy and talks to the Double Motor over Bluetooth, 50 times a second: read the sensors,
ask the policy for two arm speeds, send them. Needs: pip install legoeducation torch numpy

    python3 walk_quarter_long_lego.py check    # do this first: checks the signs below and measures MOTOR_TOP_SPEED
    python3 walk_quarter_long_lego.py          # walk

Start pose (the pose the simulation starts in): robot flat on the floor, pointing where it should walk, both arms
lying straight back along the body. The end with the axles is the front.

What the policy saw in simulation, and where it comes from here:
    how fast the body moves (3 numbers)   not measurable here -> the average value from simulation
    how fast the body turns (3)           gyroscope
    which way is down (3)                 accelerometer
    heading (2)                           the Double Motor's own yaw angle
    distance off the line (1)             not measurable here -> 0 (so it holds its heading, but cannot undo a sideways slip)
    arm angles (4), arm speeds (2)        motor encoders
    its own last output (2)               remembered (after the cut-off at -1 and +1)
The two "not measurable" ones were tested in simulation only with the walk_quarter_rl.py policy (it walked just as
straight). This policy has not been tested that way. In its film it settles about 5 cm to the left of the line.
"""
import math
import sys
import time
from collections import deque
from pathlib import Path

import legoeducation as le
import numpy as np
import torch

# ---- constants you may want to change ----------------------------------------------------------------
POLICY = Path(__file__).with_name("walk_quarter_long_policy.pt")   # from runs/walk_quarter_long_rl-20261116-224012
CARD_COLOR, CARD_SERIAL = le.LEGO_COLOR_PURPLE, "6235"    # for example le.LEGO_COLOR_PURPLE, "6235"; None = the first Double Motor found
RUN_SECONDS = 50.0              # the length of walk the policy practised

# The next block describes how YOUR robot is built. "check" tells you whether each one is right.
MOTOR_TOP_SPEED = 14.0          # rad/s an arm turns at 100 % speed, in the air ("check" measures it)
ON_LEFT, ON_RIGHT = le.MOTOR_LEFT, le.MOTOR_RIGHT   # which motor is on the robot's left / right, seen from behind
LEFT_FORWARD, RIGHT_FORWARD = -1, 1   # the sign of speed that turns that arm the way a wheel rolls forward
YAW_FACE = le.DEVICE_FACE_TOP   # the face of the Double Motor that points at the ceiling
YAW_LEFT = -1                   # -1 if the yaw angle goes down when the robot turns left (LEGO's usual), else 1
# Which way the robot's forward, left and up point, in the IMU's own (x, y, z) axes.
IMU_FORWARD, IMU_LEFT, IMU_UP = (1, 0, 0), (0, 1, 0), (0, 0, 1)
ACCEL_UP = 1                    # 1 if the accelerometer reads positive along the axis that points up, else -1
# -------------------------------------------------------------------------------------------------------

# Fixed by the simulation (walk_quarter_long_rl.py); change them only if you change them there and train again.
CONTROL_HZ = 50
MAX_MOTOR_SPEED = 2.0           # rad/s; the fastest an arm is ever asked to turn (a policy output of 1 asks for this)
TIP_ANGLE = math.radians(70.0)
IMU_UNIT = math.radians(0.1)    # the IMU reports angles in 0.1 degree and turn rates in 0.1 degree per second
IMU_AXES = np.array([IMU_FORWARD, IMU_LEFT, IMU_UP], dtype=float)


class Policy:
  """The trained network: 20 numbers in, 2 out (each cut off at -1 and +1), exactly as rsl_rl computes it."""

  def __init__(self, file):
    p = {k: v.double().numpy() for k, v in torch.load(file, map_location="cpu", weights_only=True)["actor_state_dict"].items()}
    self.mean, self.std = p["obs_normalizer._mean"][0], p["obs_normalizer._std"][0]
    self.layers = [(p[f"mlp.{i}.weight"], p[f"mlp.{i}.bias"]) for i in (0, 2, 4)]

  def __call__(self, obs):
    h = (obs - self.mean) / (self.std + 1e-2)
    for i, (weight, bias) in enumerate(self.layers):
      h = weight @ h + bias
      if i < 2:
        h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))   # ELU
    return np.clip(h, -1.0, 1.0)


class Robot:
  """The Double Motor, read and driven in the simulation's terms."""

  def __init__(self):
    self.dm = le.DoubleMotor()
    self.encoders = deque(maxlen=11)  # (time, left arm, right arm) of the last 11 sensor messages, about 0.2 s
    self.down = None
    self.owed = [0.0, 0.0]            # the part of each arm's speed (in percent) that rounding has left out so far
    self.dm.set_notification_callback(self._on_message)

  def connect(self):
    print("Connecting to Double Motor...")
    self.dm.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL, device_notification_delay=1000 // CONTROL_HZ)
    if not self.dm.connected:
      sys.exit("could not connect to the Double Motor")
    self.dm.imu_set_yaw_face(YAW_FACE)

  def _on_message(self, _message):
    left, right = self.dm.motor[ON_LEFT].position, self.dm.motor[ON_RIGHT].position
    if not (math.isnan(left) or math.isnan(right)):
      self.encoders.append((time.monotonic(), LEFT_FORWARD * math.radians(left), RIGHT_FORWARD * math.radians(right)))

  def zero(self):
    """Call this the pose "arms at angle 0, heading 0"."""
    self.dm.motor_reset_relative_position(motor=le.MOTOR_BOTH)
    self.dm.imu_reset_yaw_axis(0)
    time.sleep(0.3)
    self.encoders.clear()
    self.down = None
    self.owed = [0.0, 0.0]
    while len(self.encoders) < 2:
      time.sleep(0.01)

  def arms(self):
    """Each arm's angle (rad) and speed (rad/s), left then right; positive = the way a wheel rolls forward."""
    first, last = self.encoders[0], self.encoders[-1]
    angle = np.array(last[1:])
    return angle, (angle - np.array(first[1:])) / max(last[0] - first[0], 1e-3)

  def turn_rate(self):
    """How fast the body turns about its forward, left and up axes, rad/s."""
    imu = self.dm.imu_device
    return IMU_AXES @ np.array([imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ]) * IMU_UNIT

  def which_way_is_down(self):
    """A length-1 arrow pointing down, in the body's forward, left, up axes: (0, 0, -1) when it sits flat."""
    imu = self.dm.imu_device
    accel = IMU_AXES @ np.array([imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ], dtype=float)
    down = -ACCEL_UP * accel / max(np.linalg.norm(accel), 1e-6)
    self.down = down if self.down is None else 0.5 * self.down + 0.5 * down   # smooth out the bumps of walking
    return self.down / max(np.linalg.norm(self.down), 1e-6)

  def heading(self):
    """Rad; 0 = the way it pointed at zero(), positive = turned left."""
    return YAW_LEFT * self.dm.imu_device.yaw * IMU_UNIT

  def observation(self, policy, last_action):
    """The 20 numbers, in the order walk_quarter_long_rl.py gave them to the policy.

    The simulation's two joints are (arm on the robot's left, arm on its right), and its left one counts
    backwards: there, walking forward is a negative speed on the first joint and a positive one on the second.
    """
    angle, speed = self.arms()
    angle, speed = angle * (-1, 1), speed * (-1, 1)
    yaw = self.heading()
    return np.concatenate([
      policy.mean[0:3], self.turn_rate(), self.which_way_is_down(),
      [math.sin(yaw), math.cos(yaw), 0.0], np.sin(angle), np.cos(angle), speed, last_action,
    ])

  def drive(self, left, right):
    """Ask each arm for a speed in rad/s, positive = the way a wheel rolls forward.

    The motor takes whole percents of its top speed, and one percent is about 0.14 rad/s: coarse, when the most
    we ever ask for is 2 rad/s. So what rounding leaves out is added to the next step, and the average comes out right.
    """
    with self.dm.batch(blocking=False):   # both motors in one Bluetooth packet
      for i, (motor, sign, speed) in enumerate(((ON_LEFT, LEFT_FORWARD, left), (ON_RIGHT, RIGHT_FORWARD, right))):
        wanted = 100 * sign * speed / MOTOR_TOP_SPEED + self.owed[i]
        percent = max(-100, min(100, round(wanted)))
        self.owed[i] = wanted - percent
        self.dm.motor_run(motor=motor, speed=percent)

  def stop(self):
    if self.dm.connected:
      self.dm.motor_stop(motor=le.MOTOR_BOTH)
      # drive() does not wait for the Double Motor's replies, so some are still on their way. Disconnecting marks
      # each of those as failed ("Device disconnected unexpectedly"), and Python prints every failure nobody looked
      # at. So: give the replies a moment to arrive, then look at the ones that failed anyway.
      time.sleep(0.2)
      waiting = [reply for replies in getattr(self.dm, "_pending_responses", {}).values() for reply in replies]
      self.dm.disconnect()
      time.sleep(0.2)
      for reply in waiting:
        if reply.done() and not reply.cancelled():
          reply.exception()


def walk(robot):
  policy = Policy(POLICY)
  input("Robot flat on the floor, pointing where it should walk, both arms straight back. Then press Enter. ")
  robot.zero()
  print(f"  arms are asked for at most {MAX_MOTOR_SPEED} rad/s, which is {100 * MAX_MOTOR_SPEED / MOTOR_TOP_SPEED:.0f} % of their top speed")
  print("  second   heading (deg)   arm speeds, left and right (rad/s)")
  action = np.zeros(2)
  tipped = 0
  fastest = 0.0
  start = time.monotonic()
  for step in range(int(RUN_SECONDS * CONTROL_HZ)):
    obs = robot.observation(policy, action)
    action = policy(obs)
    robot.drive(-MAX_MOTOR_SPEED * action[0], MAX_MOTOR_SPEED * action[1])
    tipped = tipped + 1 if obs[8] > -math.cos(TIP_ANGLE) else 0
    if tipped > CONTROL_HZ // 4:
      print("tipped over, stopping")
      break
    left, right = robot.arms()[1]
    if step >= CONTROL_HZ // 4:               # the first speeds are measured over too short a time to trust
      fastest = max(fastest, abs(left), abs(right))
    if step % CONTROL_HZ == CONTROL_HZ - 1:   # once a second
      print(f"  {(step + 1) / CONTROL_HZ:5.0f}  {math.degrees(robot.heading()):11.1f}   {left:10.2f} {right:6.2f}")
    time.sleep(max(0.0, start + (step + 1) / CONTROL_HZ - time.monotonic()))
  print(f"  fastest an arm actually turned: {fastest:.2f} rad/s (in the simulation's film: 1.60)")


def check(robot):
  """Show what the script reads, so you can compare it with what you do to the robot by hand."""
  robot.zero()
  print("""
Part 1, 30 seconds. Move the robot by hand and compare (the end with the axles is the front):
  flat on the table           down = ( 0.0,  0.0, -1.0)
  nose tipped down            down: the first number goes positive
  left side tipped down       down: the second number goes positive
  while you turn it left      turning: the third number is positive;   afterwards heading is positive
  while you tip the nose down turning: the second number is positive
  turn the LEFT arm by hand the way a wheel rolls forward: the left arm angle goes up (same for the right)
If one is wrong, fix the matching constant at the top of this file and run check again.
""")
  end = time.monotonic() + 30
  while time.monotonic() < end:
    down, turning, (left, right) = robot.which_way_is_down(), robot.turn_rate(), np.degrees(robot.arms()[0])
    print(f"\r  down ({down[0]:5.2f},{down[1]:5.2f},{down[2]:5.2f})  turning ({turning[0]:5.1f},{turning[1]:5.1f},{turning[2]:5.1f}) rad/s"
          f"  heading {math.degrees(robot.heading()):6.1f} deg  arms {left:6.0f} {right:6.0f} deg ", end="", flush=True)
    time.sleep(0.1)

  input("\n\nPart 2. Hold the robot in the air so the arms can spin freely, then press Enter. ")
  for name, speeds in (("LEFT", (1.5, 0.0)), ("RIGHT", (0.0, 1.5))):
    print(f"  the {name} arm should now turn at walking speed (a quarter turn a second), the way a wheel rolls forward")
    end = time.monotonic() + 3
    while time.monotonic() < end:      # sent over and over, as in the walk, so the rounding averages out
      robot.drive(*speeds)
      time.sleep(1 / CONTROL_HZ)
  print("  both arms at full speed ...")
  robot.dm.motor_run(motor=le.MOTOR_BOTH, speed=100)
  time.sleep(1.0)                    # let them get up to speed
  before, t0 = robot.arms()[0], time.monotonic()
  time.sleep(1.5)
  speed = np.abs(robot.arms()[0] - before) / (time.monotonic() - t0)
  print(f"  top speed: left {speed[0]:.1f}, right {speed[1]:.1f} rad/s.  Set MOTOR_TOP_SPEED = {speed.mean():.1f}")


if __name__ == "__main__":
  robot = Robot()
  try:
    robot.connect()
    check(robot) if sys.argv[1:] == ["check"] else walk(robot)
  except KeyboardInterrupt:
    print("\nstopped")
  finally:
    robot.stop()
