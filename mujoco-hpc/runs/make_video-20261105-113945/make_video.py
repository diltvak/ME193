"""Record a video of a trained Ping-Pong policy. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
      --with scripts/pingpong_task/pingpong_task.py \
      --with runs/<your run>/logs/rsl_rl/ping_pong/<date folder>/model_999.pt \
      scripts/pingpong_task/make_video.py model_999.pt

It prints the highest point the ball reaches after each bounce (the goal is the start height, 1.5 m).
"""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import pingpong_task as T  # registers "Ping-Pong"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

TASK = "Ping-Pong"
STEPS = 800          # 8 seconds at 100 steps per second

if len(sys.argv) != 2:
  sys.exit("usage: make_video.py model_999.pt")
checkpoint = Path(sys.argv[1])

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(env, video_folder=".", step_trigger=lambda step: step == 0, video_length=STEPS, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

unwrapped = env.unwrapped
obs = env.get_observations()
prev_vz, apex, bounces = 0.0, 0.0, []
with torch.no_grad():
  for step in range(STEPS):
    obs = env.step(policy(obs))[0]
    z = T._ball_pos(unwrapped)[0, 2].item()
    vz = unwrapped.scene["ball"].data.root_link_lin_vel_w[0, 2].item()
    if vz > 0 and z > T.PADDLE_HOME_Z:
      apex = max(apex, z)
    if prev_vz > 0 >= vz and z > T.PADDLE_HOME_Z + 0.2:   # the ball just reached the top of a bounce
      bounces.append(apex)
      print(f"  t={(step + 1) / 100:5.2f} s   bounce {len(bounces)}: top at {apex:.2f} m (goal {T.BALL_START_Z} m)")
      apex = 0.0
    prev_vz = vz
env.close()
print(f"{len(bounces)} bounces; wrote the video (an .mp4 next to this script)")
