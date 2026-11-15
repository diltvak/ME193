"""Watch a trained Ping-Pong policy (or a dummy one). Same as mjlab's `play`, with our task registered.

    python play.py Ping-Pong --agent zero          # paddle holds still
    python play.py Ping-Pong --checkpoint-file logs/rsl_rl/ping_pong/<run>/model_999.pt
"""
import sys

import pingpong_task  # noqa: F401  (registers "Ping-Pong")
from mjlab.scripts.play import main

if __name__ == "__main__":
  sys.exit(main())
