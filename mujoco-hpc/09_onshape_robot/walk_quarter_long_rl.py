"""Teach the Onshape robot to walk straight at a slow speed for 50 seconds with reinforcement learning (mjlab), and film it.

The same as walk_quarter_rl.py (same TARGET_SPEED of 0.015 m/s, same MAX_MOTOR_SPEED of 2 rad/s), with these changes:
  - EPISODE_S is 50 seconds instead of 10. In walk_quarter_rl.py every robot started over after 10 seconds, so the
    policy never practised anything later than that, and it veered off course from about there on.
  - ITERATIONS is 2000 instead of 500. A 50 second walk is 2500 control steps and one training round is only 24, so
    more rounds are needed for every robot to finish a good number of whole walks (about 19 here).
  - FILM_SECONDS is 50 instead of 10, so the film shows the whole walk (about 0.75 m).

Run it with runIt.py from the mujoco-hpc folder (it needs a GPU node; training takes about 10 minutes):

    python3 runIt.py --gpu --time 00:45:00 --mem 16G --cpus 4 \
        --with 09_onshape_robot/onshape 09_onshape_robot/walk_quarter_long_rl.py

It trains, saves the trained policy as walk_quarter_long_policy.pt, films it as walk_quarter_long.mp4, and runIt.py
brings both home.
To film a policy you already trained, without training again, send it along and name it:

    python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 --with 09_onshape_robot/onshape \
        --with runs/walk_quarter_long_rl-<time>/walk_quarter_long_policy.pt \
        09_onshape_robot/walk_quarter_long_rl.py walk_quarter_long_policy.pt

Everything you may want to change is in the first block of constants.
"""
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

if sys.platform == "linux":
  os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  bad_orientation,
  base_ang_vel,
  base_lin_vel,
  joint_vel_rel,
  last_action,
  projected_gravity,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig

# ---- constants you may want to change ----------------------------------------------------------------
MODEL = "onshape/robot.xml"                 # your Onshape robot (the floor comes from mjlab, so not scene.xml)
LEFT, RIGHT = "left_wheel", "right_wheel"   # the two motor joints, as named in robot.xml
OUT = "walk_quarter_long.mp4"
POLICY = "walk_quarter_long_policy.pt"   # the trained policy is saved under this name

ROBOTS = 4096               # how many copies of the robot learn at the same time
ITERATIONS = 2000           # training rounds; each one is 24 control steps of every robot, then a policy update
FILM_SECONDS = 50.0         # film a whole 50 second walk

TARGET_SPEED = 0.015        # m/s; the speed the robot should hold (a quarter of the 0.06 m/s in walk_slow_rl.py)
MAX_MOTOR_SPEED = 2.0       # rad/s; the fastest a motor is ever asked to turn, in either direction (the policy's
                            # output is cut off at -1 and +1, and 1 asks for this speed)
MOTOR_KD = 0.2              # how hard a motor pushes toward the speed it was asked for (N*m per rad/s)
MOTOR_TORQUE_LIMIT = 0.3    # N*m
START_YAW = 0.4             # rad; each robot starts pointing up to this far away from straight ahead (+x)
LINE_WIDTH = 0.05           # m; how far off the line y = 0 still counts as "nearly on it"
HEADING_WIDTH = 0.3         # rad; how far off +x still counts as "nearly straight"
MAX_OFF_LINE = 0.5          # m; a robot that wanders this far from the line starts over
EPISODE_S = 50.0            # seconds one robot walks before it starts over (10 in walk_quarter_rl.py)

# What the robot is paid for, each control step. Bigger number = matters more.
REWARD_WEIGHTS = {
  "forward_speed": 1.0,     # walk along +x at TARGET_SPEED
  "stay_on_line": 1.0,      # ... and do it near the line y = 0
  "face_forward": 1.0,      # ... and do it pointing along +x
  "action_rate": -0.01,     # do not jerk the motors (a penalty, so negative)
}
# -------------------------------------------------------------------------------------------------------

CONTROL_HZ = 50             # the policy picks new motor speeds this often (physics step 0.005 s, 4 steps per pick)


# ---- the robot ------------------------------------------------------------------------------------------
def robot_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(MODEL)
  for actuator in list(spec.actuators):   # mjlab adds its own motors (below), so drop the exported ones
    spec.delete(actuator)
  if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
    spec.worldbody.first_body().add_freejoint()   # let the robot move around (same as drive.py)
  return spec


# Look at the robot once: the name of its main body, and how high to start it so it rests just above the floor.
_model = robot_spec().compile()
_data = mujoco.MjData(_model)
mujoco.mj_forward(_model, _data)
TRUNK = _model.body(1).name
START_Z = 0.005 - float((_data.geom_xpos[:, 2] - _model.geom_rbound).min())   # lowest point 5 mm up (same idea as drive.py)


def robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=robot_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, START_Z), joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no angle target, so each motor only chases the speed the policy asks for
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT), stiffness=0.0, damping=MOTOR_KD, effort_limit=MOTOR_TORQUE_LIMIT
        ),
      ),
    ),
  )


# ---- what the policy sees, and what it is paid for ----------------------------------------------------
def _yaw(env) -> torch.Tensor:
  """Heading: 0 when the robot points along +x."""
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _off_line(env) -> torch.Tensor:
  """Sideways distance from the robot's own start line, in metres."""
  return env.scene["robot"].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]


def heading_and_offset(env) -> torch.Tensor:
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw), _off_line(env) / LINE_WIDTH], dim=-1)


def motor_angles(env) -> torch.Tensor:
  """Where each arm is in its turn, as sin and cos (so a full turn brings it back to the same numbers)."""
  angles = env.scene["robot"].data.joint_pos
  return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def forward_speed(env) -> torch.Tensor:
  """1 at TARGET_SPEED along +x, 0 standing still or at twice the target, negative going backwards or faster still."""
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  return torch.clamp(1.0 - (vx - TARGET_SPEED).abs() / TARGET_SPEED, min=-1.0)


# The next two pay only while the robot is moving forward. If they paid for simply being on the line, the robot
# would learn to sit still on it (it starts there), which is safer than walking and earns almost as much.
def stay_on_line(env) -> torch.Tensor:
  """Up to 1 for moving forward on the line y = 0, falling toward 0 as the robot drifts sideways."""
  return forward_speed(env).clamp(min=0.0) * torch.exp(-torch.square(_off_line(env) / LINE_WIDTH))


def face_forward(env) -> torch.Tensor:
  """Up to 1 for moving forward while pointing along +x, falling toward 0 as the robot turns away."""
  return forward_speed(env).clamp(min=0.0) * torch.exp(-torch.square(_yaw(env) / HEADING_WIDTH))


def too_far_off_line(env) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


REWARDS = {"forward_speed": forward_speed, "stay_on_line": stay_on_line, "face_forward": face_forward,
           "action_rate": action_rate_l2}


# ---- the environment (one robot's world; mjlab runs ROBOTS of them side by side on the GPU) -----------------
def env_cfg(num_envs: int, play: bool = False) -> ManagerBasedRlEnvCfg:
  seen = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "tilt": ObservationTermCfg(func=projected_gravity),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "motor_angles": ObservationTermCfg(func=motor_angles),
    "motor_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg()},
      num_envs=num_envs,
      env_spacing=2.0,
    ),
    observations={"actor": ObservationGroupCfg(dict(seen)), "critic": ObservationGroupCfg(dict(seen))},
    actions={
      "motors": JointVelocityActionCfg(
        entity_name="robot", actuator_names=(LEFT, RIGHT), scale=MAX_MOTOR_SPEED, use_default_offset=True
      ),
    },
    events={
      "reset_pose": EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
      ),
    },
    rewards={name: RewardTermCfg(func=REWARDS[name], weight=weight) for name, weight in REWARD_WEIGHTS.items()},
    terminations={
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      "tipped_over": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(70.0)}),
      "off_line": TerminationTermCfg(func=too_far_off_line),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=TRUNK,
      distance=0.5,
      elevation=-25.0,
      azimuth=140.0,
      width=640,
      height=480,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=1.0 / (CONTROL_HZ * 4))),
    decimation=4,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """The learning method (PPO) and the size of the policy network. The defaults here are a normal starting point."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    clip_actions=1.0,               # cut the policy's output off at -1 and +1, so no motor is asked for more than MAX_MOTOR_SPEED
    experiment_name="walk_quarter_long",
    logger="tensorboard",           # the other choice, wandb, needs an account and a login
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=ITERATIONS,
  )


# ---- train ------------------------------------------------------------------------------------------------
def train(device: str) -> None:
  agent = ppo_cfg()
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg(ROBOTS), device=device), clip_actions=agent.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent), "logs", device)   # progress curves and checkpoints go to logs/
  runner.learn(num_learning_iterations=ITERATIONS, init_at_random_ep_len=True)
  runner.save(POLICY)
  env.close()
  print(f"wrote {POLICY}")


# ---- film -------------------------------------------------------------------------------------------------
def film(device: str, policy_file: str) -> None:
  agent = ppo_cfg()
  steps = int(FILM_SECONDS * CONTROL_HZ)
  env = ManagerBasedRlEnv(cfg=env_cfg(1, play=True), device=device, render_mode="rgb_array")
  env = VideoRecorder(env, video_folder="film", step_trigger=lambda step: step == 0, video_length=steps, disable_logger=True)
  env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent), None, device)
  runner.load(policy_file, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.unwrapped.scene["robot"]
  print(f"  aiming for {TARGET_SPEED} m/s, so about {TARGET_SPEED * FILM_SECONDS:.2f} m in {FILM_SECONDS:.0f} seconds")
  print(f"  motors are asked for at most {MAX_MOTOR_SPEED} rad/s")
  print("  second   x (m)   y (m)   heading (deg)   faster motor (rad/s)")
  fastest = 0.0
  obs = env.get_observations()
  with torch.no_grad():
    for step in range(steps):
      obs = env.step(policy(obs))[0]
      motor_speed = robot.data.joint_vel[0].abs().max().item()   # the faster of the two motors, right now
      fastest = max(fastest, motor_speed)
      if step % CONTROL_HZ == CONTROL_HZ - 1:   # once a second
        x, y = robot.data.root_link_pos_w[0, :2].tolist()
        print(f"  {(step + 1) / CONTROL_HZ:5.0f}  {x:7.2f} {y:7.2f}  {math.degrees(_yaw(env.unwrapped)[0].item()):11.1f}"
              f"   {motor_speed:12.2f}")
  print(f"  fastest a motor actually turned: {fastest:.2f} rad/s")
  env.close()
  videos = sorted(Path("film").glob("*.mp4"))
  if not videos:
    sys.exit("no video was recorded")
  videos[-1].replace(OUT)
  Path("film").rmdir()
  print(f"wrote {OUT}")


if __name__ == "__main__":
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"robot: {MODEL}, main body '{TRUNK}', motors {LEFT} and {RIGHT}, running on {device}")
  if len(sys.argv) > 1:
    film(device, sys.argv[1])       # a policy file was given: only film it
  else:
    train(device)
    film(device, POLICY)
