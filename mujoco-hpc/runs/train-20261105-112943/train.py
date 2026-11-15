"""Train the Ping-Pong task. Same as mjlab's own `train`, but it registers our task first.

    python train.py Ping-Pong --env.scene.num-envs 4096
"""
import sys

import pingpong_task  # noqa: F401  (registers "Ping-Pong")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
