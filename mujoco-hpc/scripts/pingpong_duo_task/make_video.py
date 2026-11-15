"""Record a video of a trained Ping-Pong-Duo policy. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
      --with scripts/pingpong_duo_task/pingpong_duo_task.py \
      --with runs/<your run>/logs/rsl_rl/ping_pong_duo/<date folder>/model_2999.pt \
      scripts/pingpong_duo_task/make_video.py model_2999.pt

It prints a line each time the ball reaches the paddle it was meant for (a completed pass).
"""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import pingpong_duo_task as T  # registers "Ping-Pong-Duo"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

TASK = "Ping-Pong-Duo"
STEPS = 1000         # 10 seconds at 100 steps per second

if len(sys.argv) != 2:
  sys.exit("usage: make_video.py model_2999.pt")
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
passes, drops, apex = 0, 0, 0.0
with torch.no_grad():
  for step in range(STEPS):
    obs, _, dones, _ = env.step(policy(obs))
    if dones[0]:     # the ball was lost and the episode restarted with a fresh drop
      drops += 1
      print(f"  t={(step + 1) / 100:5.2f} s   ball lost; new ball dropped")
      apex = 0.0
      continue
    state = T._state(unwrapped)
    apex = max(apex, T._ball_pos(unwrapped)[0, 2].item())
    if state.just_passed[0]:
      passes += 1
      catcher = T._SIDES[1 - int(state.target[0])]   # the target has already switched to the other paddle
      print(f"  t={(step + 1) / 100:5.2f} s   hit {passes}: {catcher} paddle (ball peaked at {apex:.2f} m on the way)")
      apex = 0.0

env.close()
print(f"{passes} hits on the right paddle at the right time, {drops} balls lost; wrote the video (an .mp4 next to this script)")
