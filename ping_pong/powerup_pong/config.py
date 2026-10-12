"""Tunable constants for powerup pong: difficulty, scene geometry, hit feel, minifig detection.

Powerup-specific tuning (wall size, ice duration, ...) lives at the top of each powerup's own
module in powerups/, so adding a powerup never means editing this file.
"""

import os

from swing import GREEN, RED, WHITE, YELLOW   # BGR colors shared with the rest of the project

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ping_pong/

# ---- Difficulty ----
# flight: seconds for the ball to travel from the opponent to you; window: seconds either side of
# arrival in which a swing can return it; reach: how close (in panel widths) the paddle must be to
# the ball; miss_p: chance the opponent fails to return; spread: how wide the opponent aims.
DIFFICULTIES = {
    "easy":   dict(flight=2.0, window=0.40, reach=0.22, miss_p=0.20, spread=0.22,
                   blurb="slow ball, forgiving timing"),
    "medium": dict(flight=1.4, window=0.28, reach=0.17, miss_p=0.12, spread=0.30,
                   blurb="faster ball, tighter timing"),
    "hard":   dict(flight=0.95, window=0.18, reach=0.12, miss_p=0.06, spread=0.36,
                   blurb="fast, wide shots, small window"),
}
PRACTICE = dict(flight=2.2, window=0.45, reach=0.25, miss_p=0.0, spread=0.20, blurb="")
POINT_CHOICES = (3, 5, 7, 11)

AFTER_WINDOW_FRAC = 0.6   # the hit window is shorter after the ball's arrival than before it
SWING_MEMORY_S = 0.25     # a detected swing can still return the ball this long afterward
SPEEDUP, MIN_SPEEDUP = 0.96, 0.65   # each return in a rally shortens the flight, down to this fraction
SERVE_DELAY_S = 1.2
RETURN_S = 0.8            # flight time of your return back to the opponent
RESULT_S = 1.6

# ---- Scene geometry ----
# World: X is sideways (panel widths at z = 1), Y is height above the table (panel heights at
# z = 1), z is distance from the player (the ball is on you at z = 1, behind you below that).
VP = (0.5, 0.28)          # vanishing point, as a fraction of the panel
G_OFF = 0.52              # table surface sits this far below the vanishing point at z = 1
Z_FAR, Z_NEAR, Z_MIN = 5.0, 1.0, 0.15
Z_TABLE_NEAR, Z_NET, TABLE_HALF_W, NET_H = 1.5, 3.25, 0.6, 0.07
BOUNCE_P = 0.6            # fraction of the flight at which the ball bounces on the table
BALL_R = 0.045            # ball radius in panel widths at z = 1

PADDLE_R = 0.095          # paddle radius in panel widths
PADDLE_CENTER = (0.5, 0.55)   # camera-frame point that maps to the middle of the game view
PADDLE_GAIN = 1.6         # wrist travel in the camera -> paddle travel in the game view
PADDLE_SMOOTHING = 0.5

# ---- Hit classification ----
SMASH_DPS = 400.0         # swing power (gyro peak, deg/s) at or above which a hit is a smash
EDGE_OFFSET = 0.7         # paddle-to-ball distance as a fraction of the reach beyond which it's an edge hit
EDGE_TIMING = 0.8         # swing-vs-arrival error as a fraction of the hit window beyond which it's an edge hit
POWER_WINDOW_S = 0.6      # how far back the swing power looks
HIT_STYLE = {"block": ("BLOCK", WHITE), "smash": ("SMASH!", RED), "edge": ("EDGE!", (0, 165, 255))}

# ---- Colors (BGR) ----
TABLE, TABLE_SIDE = (140, 85, 25), (90, 50, 15)
WOOD, PADDLE_RED, PADDLE_DARK = (60, 110, 170), (40, 40, 210), (20, 20, 70)
BALL_COLOR = (40, 150, 255)
MENU_BG = (45, 30, 25)

# ---- Minifigure detection (YOLO) ----
WEIGHTS_PATH = os.path.join(PARENT_DIR, "weights.pt")
YELLOW_CLASS, GREEN_CLASS, NO_FIG_CLASS = "Yellow MiniFig", "Green MiniFig", "No MiniFig"
MINIFIG_COLORS = {YELLOW_CLASS: YELLOW, GREEN_CLASS: GREEN}   # box colors on the webcam panel
MINIFIG_CONF = 0.4        # minimum YOLO confidence for a detection to count
MINIFIG_IMGSZ = 640       # image size the model was trained at
MINIFIG_CONFIRM_FRAMES = 3   # consecutive detections before a minifig "appears"
MINIFIG_REARM_S = 1.5     # a class can't trigger again this soon after it last did

# ---- Powerup HUD ----
BANNER_S = 1.6            # how long the "powerup activated" banner stays up
PRACTICE_RESET_S = 2.0    # in warm-up, a used powerup becomes ready again this long after it ends
