"""mjlab task: teach TWO ping-pong paddles to pass one ball back and forth.

Two paddles sit PADDLE_SEPARATION apart along x. Each one is the paddle from pingpong_task/ (a thin disc that
slides in x, y and z) with two extra joints so it can TILT: a level paddle can only send the ball straight up,
so to pass it sideways the paddle has to lean toward its partner. The ball is dropped above one of the paddles
(a random one each episode).

One policy drives both paddles: it sees the ball and both paddles, and outputs 10 numbers (a target for each
of the 5 joints on each paddle).

How the task knows what a "pass" is: at any moment one paddle is the TARGET, the one the ball should go to
next. It starts as the paddle the ball is dropped above. When the ball touches the target paddle, the other
paddle becomes the target. Bouncing the ball again on the same paddle changes nothing, so the only way to keep
earning reward is to send the ball across.

Importing this file registers the task "Ping-Pong-Duo" (see train.py / make_video.py next to it).
Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import action_rate_l2, joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
BALL_RADIUS = 0.04          # m (a real ping-pong ball)
BALL_MASS = 0.0027          # kg
BALL_START_Z = 1.5          # m: where the ball is dropped, and about how high a good pass should arc
BALL_START_XY = 0.05        # m; the ball starts up to this far off the centre of the paddle below it
PADDLE_SEPARATION = 1.0     # m between the two paddles' home positions (they sit at x = -half and +half)
PADDLE_RADIUS = 0.15        # m
PADDLE_HALF_THICK = 0.01    # m (so each paddle is 2 cm thick)
PADDLE_MASS = 0.3           # kg
PADDLE_HOME_Z = 0.5         # m: height of a paddle's centre when all its joints are at 0
PADDLE_RANGE_XY = 0.3       # m a paddle can slide sideways, each way
PADDLE_RANGE_Z = 0.4        # m a paddle can slide up and down, each way
PADDLE_RANGE_TILT = 0.6     # rad (about 35 degrees) a paddle can tilt, each way
ACTION_SCALE_XY = 0.25      # m: a policy output of 1.0 asks the paddle to move this far sideways from home
ACTION_SCALE_Z = 0.35       # m: ... this far up or down
ACTION_SCALE_TILT = 0.5     # rad: ... and to tilt this far
PADDLE_KP = 3000.0          # N/m   how hard a paddle's slide motors pull toward the target position
PADDLE_KD = 60.0            # N*s/m damping (about critical for this mass)
PADDLE_FORCE_LIMIT = 150.0  # N
TILT_ARMATURE = 0.005       # kg*m^2 added to each tilt joint (a motor's rotor); keeps the tiny disc stable
TILT_KP = 40.0              # N*m/rad  how hard the tilt motors pull toward the target angle
TILT_KD = 1.0               # N*m*s/rad damping (about critical with the armature)
TILT_TORQUE_LIMIT = 5.0     # N*m
CONTACT_SOLREF = "0.005 0.5"  # ball/paddle contact "timeconst dampratio": lower dampratio = bouncier
TOUCH_DISTANCE = 0.2        # m; ball centre this close to a paddle centre counts as touching that paddle
MAX_BALL_DRIFT = 0.8        # m; the episode ends if the ball gets this far outside the paddles' homes
EPISODE_S = 10.0
# -------------------------------------------------------------------------------------------------------

_SIDES = ("left", "right")
_HOME_X = (-PADDLE_SEPARATION / 2, PADDLE_SEPARATION / 2)
_SLIDES = ("x", "y", "z")
_TILTS = ("pitch", "roll")   # pitch leans the paddle toward / away from its partner, roll leans it in y
_GRAVITY = 9.81


def _ball_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint()
  body.add_geom(
    name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0], mass=BALL_MASS,
    rgba=[1.0, 0.45, 0.1, 1.0], solref=[float(v) for v in CONTACT_SOLREF.split()],
  )
  return spec


def _paddles_spec() -> mujoco.MjSpec:
  """Both paddles in one model. Joints are named left_x, left_y, left_z, left_pitch, left_roll, right_x, ..."""
  spec = mujoco.MjSpec()
  spec.compiler.degree = False   # the tilt limits below are in radians (MuJoCo's default is degrees)
  for side, home_x, rgba in zip(_SIDES, _HOME_X, ([0.35, 0.6, 1.0, 1.0], [1.0, 0.4, 0.4, 1.0])):
    body = spec.worldbody.add_body(name=f"{side}_paddle", pos=[home_x, 0, PADDLE_HOME_Z], gravcomp=1)
    # The slides come first and the hinges last, so the paddle tilts about its own centre wherever it is.
    for name, axis, r in (("x", [1, 0, 0], PADDLE_RANGE_XY), ("y", [0, 1, 0], PADDLE_RANGE_XY), ("z", [0, 0, 1], PADDLE_RANGE_Z)):
      body.add_joint(name=f"{side}_{name}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis, range=[-r, r], limited=True)
    for name, axis in (("pitch", [0, 1, 0]), ("roll", [1, 0, 0])):
      body.add_joint(
        name=f"{side}_{name}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis,
        range=[-PADDLE_RANGE_TILT, PADDLE_RANGE_TILT], limited=True, armature=TILT_ARMATURE,
      )
    body.add_geom(
      name=f"{side}_paddle_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PADDLE_RADIUS, PADDLE_HALF_THICK, 0],
      mass=PADDLE_MASS, rgba=rgba, solref=[float(v) for v in CONTACT_SOLREF.split()],
    )
  return spec


def _ball_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=_ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(_HOME_X[0], 0.0, BALL_START_Z)))


def _paddles_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_paddles_spec,
    init_state=EntityCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPdActuatorCfg(
          target_names_expr=(".*_x", ".*_y", ".*_z"), stiffness=PADDLE_KP, damping=PADDLE_KD,
          effort_limit=PADDLE_FORCE_LIMIT,
        ),
        BuiltinPdActuatorCfg(
          target_names_expr=(".*_pitch", ".*_roll"), stiffness=TILT_KP, damping=TILT_KD,
          effort_limit=TILT_TORQUE_LIMIT,
        ),
      ),
    ),
  )


# ---- where things are, and which paddle the ball should go to next -------------------------------------


def _ball_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position in this env's own frame (the env's origin subtracted), metres. Shape [num_envs, 3]."""
  return env.scene["ball"].data.root_link_pos_w - env.scene.env_origins


def _ball_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["ball"].data.root_link_lin_vel_w


class _PassState:
  """What the task remembers between steps: which paddle is the target, and how many passes so far."""

  def __init__(self, env: ManagerBasedRlEnv):
    paddles = env.scene["paddles"]
    # Columns of joint_pos holding each paddle's x, y, z slides: [[left x, y, z], [right x, y, z]].
    self.slide_ids = [
      paddles.find_joints([f"{side}_{n}" for n in _SLIDES], preserve_order=True)[0] for side in _SIDES
    ]
    self.home = torch.tensor([[x, 0.0, PADDLE_HOME_Z] for x in _HOME_X], device=env.device)   # [2, 3]
    self.target = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)   # 0 = left, 1 = right
    self.passes = torch.zeros(env.num_envs, device=env.device)                     # this episode
    self.just_passed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.step = env.common_step_counter


def _state(env: ManagerBasedRlEnv) -> _PassState:
  """The pass state, brought up to date for this step (the work is done once per step, however often called)."""
  s = getattr(env, "_pass_state", None)
  if s is None:
    s = env._pass_state = _PassState(env)
  if s.step != env.common_step_counter:
    s.step = env.common_step_counter
    ball = _ball_pos(env)
    touching_target = torch.linalg.norm(ball - _paddle_pos(env, s.target, s), dim=-1) < TOUCH_DISTANCE
    s.just_passed = touching_target
    s.passes += touching_target
    s.target = torch.where(touching_target, 1 - s.target, s.target)   # the other paddle is up next
  return s


def _paddle_pos(env: ManagerBasedRlEnv, which: int | torch.Tensor, s: _PassState | None = None) -> torch.Tensor:
  """Centre of a paddle in the env's frame, [num_envs, 3]. `which` is 0 (left), 1 (right) or one index per env."""
  s = s or _state(env)
  q = env.scene["paddles"].data.joint_pos
  both = torch.stack([q[:, ids] for ids in s.slide_ids], dim=1) + s.home    # [num_envs, 2, 3]
  if isinstance(which, int):
    return both[:, which]
  return both[torch.arange(env.num_envs, device=env.device), which]


def _landing_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Where the ball will come down to paddle height if nothing touches it (simple projectile motion). [num_envs, 2]."""
  pos, vel = _ball_pos(env), _ball_vel(env)
  drop = torch.clamp(pos[:, 2] - (PADDLE_HOME_Z + BALL_RADIUS), min=0.0)
  t = (vel[:, 2] + torch.sqrt(torch.square(vel[:, 2]) + 2 * _GRAVITY * drop)) / _GRAVITY   # time until it gets there
  return pos[:, :2] + vel[:, :2] * t.unsqueeze(-1)


def reset_ball_above_random_paddle(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Start of an episode: put the ball above one of the paddles (a coin flip); that paddle is the first target."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  s = _state(env)
  n = len(env_ids)
  side = torch.randint(0, 2, (n,), device=env.device)
  pos = s.home[side].clone()
  pos[:, :2] += (torch.rand(n, 2, device=env.device) * 2 - 1) * BALL_START_XY
  pos[:, 2] = BALL_START_Z
  pose = torch.zeros(n, 7, device=env.device)
  pose[:, :3] = pos + env.scene.env_origins[env_ids]
  pose[:, 3] = 1.0   # quaternion (w, x, y, z): no rotation
  ball = env.scene["ball"]
  ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  ball.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
  s.target[env_ids] = side
  s.passes[env_ids] = 0
  s.just_passed[env_ids] = False


# ---- observations, rewards, terminations -------------------------------------------------------------


def ball_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[ball position (3), velocity (3), position relative to the left paddle (3), to the right paddle (3)]."""
  pos = _ball_pos(env)
  return torch.cat([pos, _ball_vel(env), pos - _paddle_pos(env, 0), pos - _paddle_pos(env, 1)], dim=-1)


def target_side(env: ManagerBasedRlEnv) -> torch.Tensor:
  """-1 when the ball should go to the left paddle next, +1 for the right. Shape [num_envs, 1]."""
  return (_state(env).target.float() * 2 - 1).unsqueeze(-1)


def aim(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 when the ball is on course to come down on the target paddle's home, falling off with the miss distance."""
  s = _state(env)
  miss = torch.linalg.norm(_landing_xy(env) - s.home[s.target][:, :2], dim=-1)
  return 1.0 / (1.0 + torch.square(miss / 0.3))     # half the reward when it would land 0.3 m off


def be_there(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 when the target paddle is right where the ball will come down (as near as it can reach), less when off."""
  s = _state(env)
  home = s.home[s.target][:, :2]
  spot = home + torch.clamp(_landing_xy(env) - home, -ACTION_SCALE_XY, ACTION_SCALE_XY)
  off = torch.linalg.norm(_paddle_pos(env, s.target, s)[:, :2] - spot, dim=-1)
  return 1.0 / (1.0 + torch.square(off / 0.1))


def pass_made(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 on the step the ball reaches the target paddle (a completed pass), else 0."""
  return _state(env).just_passed.float()


def ball_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 when the ball is at the start height, falling off smoothly above and below it: rewards high, slow arcs."""
  return torch.exp(-torch.square((_ball_pos(env)[:, 2] - BALL_START_Z) / 0.5))


def ball_lost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """True when the ball has dropped below both paddles (missed) or flown too far away."""
  pos = _ball_pos(env)
  lowest = torch.minimum(_paddle_pos(env, 0)[:, 2], _paddle_pos(env, 1)[:, 2])
  below = pos[:, 2] < lowest - 0.05
  wide = (pos[:, 0].abs() > PADDLE_SEPARATION / 2 + MAX_BALL_DRIFT) | (pos[:, 1].abs() > MAX_BALL_DRIFT)
  return below | wide


# ---- environment ----------------------------------------------------------------------------------------


def pingpong_duo_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  paddles = SceneEntityCfg("paddles")
  actor_terms = {
    "ball": ObservationTermCfg(func=ball_state),
    "target_side": ObservationTermCfg(func=target_side),
    "paddle_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": paddles}),
    "paddle_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": paddles}),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "paddles": JointPositionActionCfg(
      entity_name="paddles",
      actuator_names=(".*",),
      scale={
        ".*_x": ACTION_SCALE_XY, ".*_y": ACTION_SCALE_XY, ".*_z": ACTION_SCALE_Z,
        ".*_pitch": ACTION_SCALE_TILT, ".*_roll": ACTION_SCALE_TILT,
      },
      use_default_offset=True,
    ),
  }

  events = {
    "reset_ball": EventTermCfg(func=reset_ball_above_random_paddle, mode="reset"),
  }

  # Each weight is "per second" (mjlab multiplies by the 0.01 s step), so pass_made's 100 is +1 per pass.
  rewards = {
    "aim": RewardTermCfg(func=aim, weight=1.0),
    "be_there": RewardTermCfg(func=be_there, weight=0.5),
    "pass_made": RewardTermCfg(func=pass_made, weight=100.0),
    "ball_height": RewardTermCfg(func=ball_height, weight=0.5),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "ball_lost": TerminationTermCfg(func=ball_lost),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"ball": _ball_cfg(), "paddles": _paddles_cfg()},
      num_envs=1,
      env_spacing=0.0,   # every env is its own copy of the world; all of them put the paddles at the same
                         # place, so the ball must start there too (a grid would move only the ball)
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.0, 0.0, 0.9),
      distance=3.8,
      elevation=-6.0,
      azimuth=90.0,
      max_extra_envs=0,    # envs all sit on top of each other (see env_spacing), so draw just one
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    decimation=5,          # 5 physics steps of 2 ms = the policy acts every 10 ms (100 Hz)
    episode_length_s=1e10 if play else EPISODE_S,
  )


def pingpong_duo_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(256, 256), activation="elu", obs_normalization=True),
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
    experiment_name="ping_pong_duo",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3000,
  )


register_mjlab_task(
  task_id="Ping-Pong-Duo",
  env_cfg=pingpong_duo_env_cfg(),
  play_env_cfg=pingpong_duo_env_cfg(play=True),
  rl_cfg=pingpong_duo_ppo_runner_cfg(),
)
