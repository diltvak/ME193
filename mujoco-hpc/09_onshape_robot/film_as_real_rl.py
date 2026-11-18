"""Film the policy from walk_quarter_long_rl.py in simulation, but under the conditions of the real LEGO robot.

Nothing is trained here. The simulated robot and its world are exactly those of walk_quarter_long_rl.py (this script
imports that file), and the policy is the one you already trained. What changes is what goes into the policy and
what happens to what comes out, copied from SillyWalks/walk_quarter_long_lego.py:

  no body speed      the three "how fast the body moves" numbers are replaced by their average from training
  no off-line        the "distance off the line" number is always 0
  slow arm speed     arm speed is the change in arm angle over the last 0.2 s, not the speed right now
  smoothed tilt      "which way is down" is averaged with its previous value, as the lego script does
  whole percents     each arm speed is rounded to a whole percent of MOTOR_TOP_SPEED (the rest is carried over)
  late commands      a command reaches the motors DELAY_STEPS control steps after the sensors were read
  strong motor       not a sensor matter: the simulated motor is made to reach the speed it is asked for. As
                     trained, it settles about 0.5 rad/s short (asked for 2.0, it turns at 1.5). The real motor has
                     its own speed controller and probably does not fall short like that. This one is a guess.

It does two things:
  1. A table. For each line above on its own, and for all of them together, ROBOTS robots walk for 50 seconds
     and the script counts how many tipped over or wandered off, and when. The line that makes the simulation
     fail the way your real robot fails (veers left, tips at 15-25 s) is the one to fix in training.
  2. A film of one robot under FILM_CASE, with the same once-a-second printout as walk_quarter_long_rl.py.

Run it with runIt.py from the mujoco-hpc folder (about 10 minutes):

    python3 runIt.py --gpu --time 00:30:00 --mem 16G --cpus 4 \
        --with 09_onshape_robot/onshape --with 09_onshape_robot/walk_quarter_long_rl.py \
        --with runs/walk_quarter_long_rl-20261116-224012/walk_quarter_long_policy.pt \
        09_onshape_robot/film_as_real_rl.py

Not copied from the real robot (so the simulation can still be kinder than reality): sensor noise, heading drift,
the accelerometer feeling the robot's own jolts, uneven Bluetooth timing, and the floor's true grip.
"""
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import walk_quarter_long_rl as rl   # the training script; sent along with --with
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

# ---- constants you may want to change ----------------------------------------------------------------
OUT = "walk_quarter_long_as_real.mp4"
SECONDS = 50.0              # length of every walk, in the table and in the film
ROBOTS = 64                 # robots per line of the table
START_YAW = 0.05            # rad; how far off straight a robot starts (about 3 degrees: aiming by hand).
                            # Training used 0.4. With 0, all the robots of one line would do exactly the same thing.

MOTOR_TOP_SPEED = 14.0      # rad/s; the same number as in walk_quarter_long_lego.py ("check" measures it)
DELAY_STEPS = 2             # control steps (0.02 s each) between reading the sensors and the motors reacting
ARM_SPEED_STEPS = 10        # arm speed is measured over this many control steps (0.2 s in the lego script)
STRONG_MOTOR_KD = 1.0       # MOTOR_KD for "strong motor" (training used 0.2); asked for 2.0 rad/s it reaches 1.9

SENSORS_AND_COMMANDS = ("no body speed", "no off-line", "slow arm speed", "smoothed tilt", "whole percents",
                        "late commands")
# The lines of the table: a name, and which changes are switched on for it.
CASES = {
  "as trained": (),
  **{name: (name,) for name in SENSORS_AND_COMMANDS},
  "strong motor": ("strong motor",),
  "like the real robot": SENSORS_AND_COMMANDS,
  "like the real robot + strong motor": SENSORS_AND_COMMANDS + ("strong motor",),
}
FILM_CASE = "like the real robot"   # which line of the table to film
# -------------------------------------------------------------------------------------------------------

DT = 1.0 / rl.CONTROL_HZ
# Where each thing sits among the 20 numbers the policy sees (the order of "seen" in walk_quarter_long_rl.py).
BODY_SPEED, TILT, OFF_LINE, ARM_SPEED = slice(0, 3), slice(6, 9), 11, slice(16, 18)
LAST_ACTION = slice(18, 20)
TRAINED_MOTOR_KD = rl.MOTOR_KD


class RealRobot:
  """Stands between the simulation and the policy, and makes each side look like the real robot to the other."""

  def __init__(self, env, on, average_body_speed):
    self.on = on
    self.robot = env.unwrapped.scene["robot"]
    self.average_body_speed = average_body_speed
    n, device = env.num_envs, env.unwrapped.device
    self.rows = torch.arange(n, device=device)
    self.arm_angles = torch.zeros(n, ARM_SPEED_STEPS + 1, 2, device=device)   # the last 11 readings, newest last
    self.readings = torch.zeros(n, dtype=torch.long, device=device)           # how many of them are real so far
    self.down = torch.zeros(n, 3, device=device)
    self.owed = torch.zeros(n, 2, device=device)       # the part of each arm's speed (in percent) rounding left out
    self.last_action = torch.zeros(n, 2, device=device)
    self.on_their_way = [torch.zeros(n, 2, device=device) for _ in range(DELAY_STEPS)]   # commands not yet arrived
    self.start_over(torch.ones(n, dtype=torch.bool, device=device), None)

  def start_over(self, who, obs):
    """Forget everything about the robots in `who` (they were just put back at the start)."""
    self.arm_angles[who] = self.robot.data.joint_pos[who].unsqueeze(1)
    self.readings[who] = 0
    self.owed[who] = 0.0
    self.last_action[who] = 0.0
    for command in self.on_their_way:
      command[who] = 0.0
    if obs is not None:
      self.down[who] = obs[who, TILT]

  def observe(self, true_obs):
    """The 20 numbers as the real robot would report them."""
    obs = true_obs.clone()
    if self.readings.max() == 0 and self.down.abs().sum() == 0:   # very first call
      self.down = obs[:, TILT].clone()

    self.arm_angles = torch.cat([self.arm_angles[:, 1:], self.robot.data.joint_pos.unsqueeze(1)], dim=1)
    if "slow arm speed" in self.on:
      back = self.readings.clamp(max=ARM_SPEED_STEPS)   # how many steps back the oldest real reading is
      oldest = self.arm_angles[self.rows, ARM_SPEED_STEPS - back]
      obs[:, ARM_SPEED] = (self.arm_angles[:, -1] - oldest) / (back.clamp(min=1).unsqueeze(1) * DT)
    self.readings += 1

    if "no body speed" in self.on:
      obs[:, BODY_SPEED] = self.average_body_speed
    if "no off-line" in self.on:
      obs[:, OFF_LINE] = 0.0
    if "smoothed tilt" in self.on:
      self.down = 0.5 * self.down + 0.5 * obs[:, TILT]
      obs[:, TILT] = self.down / self.down.norm(dim=1, keepdim=True).clamp(min=1e-6)
    obs[:, LAST_ACTION] = self.last_action   # what the policy itself said, not what rounding and delay made of it
    return obs

  def command(self, action):
    """What the motors are asked for this step, given what the policy just said (both in policy units, -1 to 1)."""
    self.last_action = action
    if "whole percents" in self.on:
      wanted = 100 * action * rl.MAX_MOTOR_SPEED / MOTOR_TOP_SPEED + self.owed
      percent = wanted.round().clamp(-100, 100)
      self.owed = wanted - percent
      action = percent * MOTOR_TOP_SPEED / 100 / rl.MAX_MOTOR_SPEED
    if "late commands" in self.on and DELAY_STEPS > 0:
      self.on_their_way.append(action.clone())
      action = self.on_their_way.pop(0)
    return action


def actor_numbers(obs):
  """The policy's 20 numbers, out of whatever the environment hands back."""
  return obs if isinstance(obs, torch.Tensor) else obs["actor"]


def with_actor_numbers(obs, numbers):
  if isinstance(obs, torch.Tensor):
    return numbers
  obs["actor"] = numbers
  return obs


def walk(device, policy_file, name, robots, film):
  """Walk `robots` robots for SECONDS under the case `name`. Returns one line of results for the table."""
  on = CASES[name]
  rl.START_YAW = START_YAW
  rl.MOTOR_KD = STRONG_MOTOR_KD if "strong motor" in on else TRAINED_MOTOR_KD
  agent = rl.ppo_cfg()
  steps = int(SECONDS * rl.CONTROL_HZ)
  env = ManagerBasedRlEnv(cfg=rl.env_cfg(robots, play=True), device=device, render_mode="rgb_array" if film else None)
  if film:
    env = VideoRecorder(env, video_folder="film", step_trigger=lambda step: step == 0, video_length=steps, disable_logger=True)
  # The policy's output is cut off at -1 and +1 below, by hand. The wrapper's own cut-off is set out of the way,
  # because a speed rounded up to the next whole percent can be a little more than 1.
  env = RslRlVecEnvWrapper(env, clip_actions=100.0)
  runner = MjlabOnPolicyRunner(env, asdict(agent), None, device)
  runner.load(policy_file, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  saved = torch.load(policy_file, map_location=device, weights_only=True)["actor_state_dict"]
  average_body_speed = saved["obs_normalizer._mean"][0, BODY_SPEED].float()

  scene = env.unwrapped.scene
  robot = scene["robot"]
  real = RealRobot(env, on, average_body_speed)
  walking = torch.ones(robots, dtype=torch.bool, device=device)   # False once a robot has failed for the first time
  failed_at = torch.full((robots,), float("nan"), device=device)
  tipped = torch.zeros(robots, dtype=torch.bool, device=device)
  arm_speed_sum = torch.zeros(robots, device=device)

  if film:
    print(f"\nfilming '{name}'")
    print("  second   x (m)   y (m)   heading (deg)   tilt (deg)   arm speeds (rad/s)")
  obs = env.get_observations()
  with torch.no_grad():
    for step in range(steps):
      true_obs = actor_numbers(obs)
      off_line = (robot.data.root_link_pos_w[:, 1] - scene.env_origins[:, 1]).abs()
      action = policy(with_actor_numbers(obs, real.observe(true_obs))).clamp(-1.0, 1.0)
      result = env.step(real.command(action))
      obs, done = result[0], result[2].bool()
      arm_speed_sum += robot.data.joint_vel.abs().mean(dim=1) * walking

      new = done & walking   # the environment has already put these robots back at the start
      failed_at[new] = (step + 1) * DT
      tipped[new] = off_line[new] < rl.MAX_OFF_LINE - 0.02   # not at the edge of the lane, so it must have tipped
      walking &= ~done
      if done.any():
        real.start_over(done, actor_numbers(obs))
        if film:
          print(f"  --- {'tipped over' if tipped[0] else 'wandered off the line'} at {(step + 1) * DT:.1f} s;"
                f" the robot starts over ---")

      if film and step % rl.CONTROL_HZ == rl.CONTROL_HZ - 1:   # once a second
        x, y = (robot.data.root_link_pos_w[0, :2] - scene.env_origins[0, :2]).tolist()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -actor_numbers(obs)[0, 8].item()))))
        left, right = robot.data.joint_vel[0].tolist()
        print(f"  {(step + 1) * DT:5.0f}  {x:7.2f} {y:7.2f}  {math.degrees(rl._yaw(env.unwrapped)[0].item()):11.1f}"
              f"  {tilt:10.1f}   {left:8.2f} {right:6.2f}")

  xy = robot.data.root_link_pos_w[:, :2] - scene.env_origins[:, :2]
  yaw = rl._yaw(env.unwrapped)
  n = int(walking.sum())

  def average(values):
    return values[walking].mean().item() if n else float("nan")

  line = {
    "name": name,
    "whole": n,
    "tipped": int((~walking & tipped).sum()),
    "off": int((~walking & ~tipped).sum()),
    "when": failed_at[~walking].median().item() if n < robots else float("nan"),
    "x": average(xy[:, 0]),
    "y": average(xy[:, 1]),
    "heading": math.degrees(average(yaw)),
    "arm": average(arm_speed_sum / steps),
  }
  env.close()
  if film:
    videos = sorted(Path("film").glob("*.mp4"))
    if not videos:
      sys.exit("no video was recorded")
    videos[-1].replace(OUT)
    Path("film").rmdir()
    print(f"wrote {OUT}")
  return line


if __name__ == "__main__":
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  policy_file = sys.argv[1] if len(sys.argv) > 1 else rl.POLICY
  print(f"policy: {policy_file}, running on {device}")

  lines = []
  for name in CASES:
    print(f"walking {ROBOTS} robots: {name}", flush=True)
    lines.append(walk(device, policy_file, name, ROBOTS, film=False))
  walk(device, policy_file, FILM_CASE, 1, film=True)

  width = max(len(name) for name in CASES)
  print(f"\n{ROBOTS} robots each, {SECONDS:.0f} seconds. The last four columns are averages over the robots that walked"
        f" the whole time\n(y > 0 and heading > 0 mean: to the left).\n")
  print(f"  {'':{width}}  whole walk  tipped  wandered off  failed at (s)   x (m)   y (m)  heading (deg)  arm speed (rad/s)")
  for r in lines:
    print(f"  {r['name']:{width}}  {r['whole']:10d}  {r['tipped']:6d}  {r['off']:12d}  {r['when']:13.1f}"
          f"  {r['x']:6.2f}  {r['y']:6.2f}  {r['heading']:13.1f}  {r['arm']:17.2f}")
