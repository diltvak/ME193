"""mjlab task: teach a ping-pong paddle to bounce a ball back up to the height it was dropped from.

The ball starts at BALL_START_Z (a few cm off-centre, random) and falls onto a paddle directly beneath it:
a thin disc (a short cylinder) that can slide in x, y and z. The policy sees the ball and the paddle and
outputs a target position for the paddle in each direction. A still paddle only gets the ball back to ~0.6 m
(the contact soaks up most of the energy), so the paddle has to learn to swing up into the ball.

Importing this file registers the task "Ping-Pong" (see train.py / make_video.py next to it).
Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import action_rate_l2, joint_pos_rel, joint_vel_rel, last_action, reset_root_state_uniform, time_out
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
BALL_START_Z = 1.5          # m: where the ball is dropped, and the height the paddle should send it back to
BALL_START_XY = 0.05        # m; the ball starts up to this far off-centre in x and y
PADDLE_RADIUS = 0.15        # m
PADDLE_HALF_THICK = 0.01    # m (so the paddle is 2 cm thick)
PADDLE_MASS = 0.3           # kg
PADDLE_HOME_Z = 0.5         # m: height of the paddle's centre when all joints are at 0
PADDLE_RANGE_XY = 0.3       # m the paddle can slide sideways, each way
PADDLE_RANGE_Z = 0.4        # m the paddle can slide up and down, each way
ACTION_SCALE_XY = 0.25      # m: a policy output of 1.0 asks the paddle to move this far sideways from home
ACTION_SCALE_Z = 0.35       # m: ... and this far up or down
PADDLE_KP = 3000.0          # N/m   how hard the paddle's motor pulls toward the target position
PADDLE_KD = 60.0            # N*s/m damping (about critical for this mass)
PADDLE_FORCE_LIMIT = 150.0  # N
CONTACT_SOLREF = "0.005 0.5"  # ball/paddle contact "timeconst dampratio": lower dampratio = bouncier
MAX_BALL_DRIFT = 0.8        # m; the episode ends if the ball gets this far sideways from the paddle's home
EPISODE_S = 8.0
# -------------------------------------------------------------------------------------------------------

_BALL = SceneEntityCfg("ball")
_PADDLE = SceneEntityCfg("paddle")
_JOINTS = ("paddle_x", "paddle_y", "paddle_z")


def _ball_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint()
  body.add_geom(
    name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0], mass=BALL_MASS,
    rgba=[1.0, 0.45, 0.1, 1.0], solref=[float(v) for v in CONTACT_SOLREF.split()],
  )
  return spec


def _paddle_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="paddle", pos=[0, 0, PADDLE_HOME_Z], gravcomp=1)  # gravcomp: no sag
  for name, axis, r in (
    ("paddle_x", [1, 0, 0], PADDLE_RANGE_XY),
    ("paddle_y", [0, 1, 0], PADDLE_RANGE_XY),
    ("paddle_z", [0, 0, 1], PADDLE_RANGE_Z),
  ):
    body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis, range=[-r, r], limited=True)
  body.add_geom(
    name="paddle_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PADDLE_RADIUS, PADDLE_HALF_THICK, 0],
    mass=PADDLE_MASS, rgba=[0.9, 0.9, 0.95, 1.0], solref=[float(v) for v in CONTACT_SOLREF.split()],
  )
  return spec


def _ball_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=_ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_START_Z)))


def _paddle_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_paddle_spec,
    init_state=EntityCfg.InitialStateCfg(joint_pos={n: 0.0 for n in _JOINTS}, joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPdActuatorCfg(
          target_names_expr=_JOINTS, stiffness=PADDLE_KP, damping=PADDLE_KD, effort_limit=PADDLE_FORCE_LIMIT
        ),
      ),
    ),
  )


# ---- observations, rewards, terminations -------------------------------------------------------------


def _ball_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position in this env's own frame (the env's origin subtracted), metres."""
  return env.scene["ball"].data.root_link_pos_w - env.scene.env_origins


def _paddle_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Paddle centre in this env's own frame: home position + how far each slide joint has moved."""
  q = env.scene["paddle"].data.joint_pos
  return q + torch.tensor([0.0, 0.0, PADDLE_HOME_Z], device=q.device)


def ball_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[ball position (3), ball velocity (3), ball position relative to the paddle (3)]. Shape [num_envs, 9]."""
  pos = _ball_pos(env)
  vel = env.scene["ball"].data.root_link_lin_vel_w
  return torch.cat([pos, vel, pos - _paddle_pos(env)], dim=-1)


def ball_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 when the ball is at the start height, falling off smoothly (a bump 0.5 m wide) above and below it."""
  return torch.exp(-torch.square((_ball_pos(env)[:, 2] - BALL_START_Z) / 0.5))


def paddle_under_ball(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalty for the paddle's sideways distance from the ball (squared, metres)."""
  return -torch.sum(torch.square((_ball_pos(env) - _paddle_pos(env))[:, :2]), dim=-1)


def ball_drift(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalty for the ball's sideways distance from the start line."""
  return -torch.sum(torch.square(_ball_pos(env)[:, :2]), dim=-1)


def ball_lost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """True when the ball has dropped below the paddle's surface (missed) or flown too far sideways."""
  pos = _ball_pos(env)
  below = pos[:, 2] < _paddle_pos(env)[:, 2] - 0.05
  wide = torch.linalg.norm(pos[:, :2], dim=-1) > MAX_BALL_DRIFT
  return below | wide


# ---- environment ----------------------------------------------------------------------------------------


def pingpong_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "ball": ObservationTermCfg(func=ball_state),
    "paddle_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": _PADDLE}),
    "paddle_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": _PADDLE}),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "paddle": JointPositionActionCfg(
      entity_name="paddle",
      actuator_names=_JOINTS,
      scale={"paddle_x": ACTION_SCALE_XY, "paddle_y": ACTION_SCALE_XY, "paddle_z": ACTION_SCALE_Z},
      use_default_offset=True,
    ),
  }

  events = {
    "reset_ball": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-BALL_START_XY, BALL_START_XY), "y": (-BALL_START_XY, BALL_START_XY)},
        "velocity_range": {},
        "asset_cfg": _BALL,
      },
    ),
  }

  rewards = {
    "ball_height": RewardTermCfg(func=ball_height, weight=1.0),
    "paddle_under_ball": RewardTermCfg(func=paddle_under_ball, weight=1.0),
    "ball_drift": RewardTermCfg(func=ball_drift, weight=0.5),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "ball_lost": TerminationTermCfg(func=ball_lost),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"ball": _ball_cfg(), "paddle": _paddle_cfg()},
      num_envs=1,
      env_spacing=0.0,   # every env is its own copy of the world; all of them put the paddle at the origin,
                         # so the ball must start there too (the default grid would move only the ball)
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.0, 0.0, 0.9),
      distance=3.2,
      elevation=-6.0,
      azimuth=90.0,
      max_extra_envs=0,    # envs all sit on top of each other (see env_spacing), so draw just one
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    decimation=5,          # 5 physics steps of 2 ms = the policy acts every 10 ms (100 Hz)
    episode_length_s=1e10 if play else EPISODE_S,
  )


def pingpong_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu", obs_normalization=True),
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
    experiment_name="ping_pong",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )


register_mjlab_task(
  task_id="Ping-Pong",
  env_cfg=pingpong_env_cfg(),
  play_env_cfg=pingpong_env_cfg(play=True),
  rl_cfg=pingpong_ppo_runner_cfg(),
)
