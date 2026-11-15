"""Train the Drive-Straight task. Same as mjlab's own `train`, but it registers our task first.

    python train.py Drive-Straight --env.scene.num-envs 4096
"""
import sys

import drive_task  # noqa: F401  (registers "Drive-Straight")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
