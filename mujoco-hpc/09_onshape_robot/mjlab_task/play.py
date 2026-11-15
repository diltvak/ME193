"""Watch a trained Drive-Straight policy (or a dummy one). Same as mjlab's `play`, with our task registered.

    python play.py Drive-Straight --agent zero          # wheels do nothing: checks the robot loads
    python play.py Drive-Straight --agent random        # random wheel speeds
    python play.py Drive-Straight --checkpoint-file logs/rsl_rl/drive_straight/<run>/model_499.pt
"""
import sys

import drive_task  # noqa: F401  (registers "Drive-Straight")
from mjlab.scripts.play import main

if __name__ == "__main__":
  sys.exit(main())
