"""Yellow minifig powerup: a brick wall rises at the end of the table and returns the ball for you."""

import random

import cv2

from powerup_pong.config import TABLE_HALF_W, YELLOW_CLASS, Z_TABLE_NEAR
from powerup_pong.geometry import to_px
from powerup_pong.powerups.base import Powerup

WALL_Z = Z_TABLE_NEAR          # the wall stands on the player's end of the table
WALL_H = 0.22                  # height above the table
WALL_HALF_W = TABLE_HALF_W + 0.06
ROWS, COLS = 4, 11
BUILD_S = 0.5                  # time for the wall to rise
CRUMBLE_S = 1.0                # time for it to fall apart after returning the ball
BRICK_COLORS = ((45, 70, 175), (55, 85, 190), (40, 60, 160))
MORTAR = (160, 170, 175)


class BrickWall(Powerup):
    name = "Brick wall"
    hud_name = "WALL"
    trigger_class = YELLOW_CLASS
    key = "y"
    color = (0, 215, 255)
    banner = "automatic return"
    fade_s = CRUMBLE_S

    def __init__(self):
        super().__init__()
        rng = random.Random(3)
        # Fixed per-brick randomness, so the wall doesn't shimmer from frame to frame.
        self.shade = [[rng.choice(BRICK_COLORS) for _ in range(COLS + 1)] for _ in range(ROWS)]
        self.fall = [[rng.random() for _ in range(COLS + 1)] for _ in range(ROWS)]

    def takes_over_returns(self):
        return self.state == "active"

    def intercept(self, game, now, world):
        if self.state != "active" or world[2] > WALL_Z:
            return False
        game.auto_return(now, world)
        self.finish(now)
        return True

    def draw_scene(self, img, game, now):
        if self.state in ("ready", "used"):
            return
        h, w = img.shape[:2]
        built = min(1.0, (now - self.t0) / BUILD_S) * ROWS
        crumble = (now - self.t_end) / CRUMBLE_S if self.state == "fading" else 0.0
        bw, bh = 2 * WALL_HALF_W / COLS, WALL_H / ROWS
        for r in range(ROWS):
            if r >= built:
                break
            for c in range(COLS + 1):
                if crumble > self.fall[r][c]:
                    continue                                   # this brick has fallen away
                x0 = -WALL_HALF_W + (c - 0.5 * (r % 2)) * bw
                x1 = x0 + bw
                x0, x1 = max(x0, -WALL_HALF_W), min(x1, WALL_HALF_W)
                if x1 - x0 < 1e-3:
                    continue
                drop = crumble * crumble * self.fall[r][c] * 0.3
                y0 = r * bh - drop
                (px0, py1), (px1, py0) = to_px(x0, y0 + bh, WALL_Z, w, h), to_px(x1, y0, WALL_Z, w, h)
                cv2.rectangle(img, (px0, py1), (px1, py0), self.shade[r][c], -1)
                cv2.rectangle(img, (px0, py1), (px1, py0), MORTAR, 1, cv2.LINE_AA)
