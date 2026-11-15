"""Train the Ping-Pong-Duo task (two paddles passing one ball). Same as mjlab's own `train`, with our task registered.

    python train.py Ping-Pong-Duo --env.scene.num-envs 4096
"""
import sys

import pingpong_duo_task  # noqa: F401  (registers "Ping-Pong-Duo")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
