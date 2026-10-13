"""Green minifig powerup: an ice storm takes over the arena and slows the ball for the next 3 returns."""

import math

import cv2
import numpy as np

from powerup_pong.config import BALL_R, GREEN_CLASS
from powerup_pong.geometry import proj
from powerup_pong.powerups.base import Powerup

HITS = 3                       # returns the storm lasts for
SLOW_FACTOR = 1.6              # flight-time multiplier for balls launched during the storm
RAMP_S = 0.6                   # time for the storm to roll in
FADE_S = 1.2                   # time for it to clear after the last hit
SNOWFLAKES = 110
ICE_TINT, FROST, SNOW = (255, 215, 150), (255, 240, 220), (255, 255, 255)   # BGR


class IceStorm(Powerup):
    name = "Ice storm"
    hud_name = "ICE"
    trigger_class = GREEN_CLASS
    key = "g"
    color = (0, 220, 0)
    banner = f"ball slowed for the next {HITS} hits"
    opponent_reaction = "frozen"
    opponent_line = "Brrr... so cold!"
    fade_s = FADE_S

    def __init__(self):
        super().__init__()
        rng = np.random.RandomState(7)
        self.flakes = rng.rand(SNOWFLAKES, 5)   # x, y, fall speed, drift phase, size
        self._frost_key, self._frost = None, None

    def reset(self):
        super().reset()
        self.hits_left = HITS

    def activate(self, game, now):
        super().activate(game, now)
        self.hits_left = HITS
        game.slow_ball(SLOW_FACTOR, now)   # a ball already in flight slows down too

    def flight_multiplier(self):
        return SLOW_FACTOR if self.state == "active" else 1.0

    def on_return(self, game, now):
        if self.state == "active":
            self.hits_left -= 1
            if self.hits_left <= 0:
                self.finish(now)

    def hud_detail(self):
        return f"{self.hits_left} hits left" if self.state == "active" else super().hud_detail()

    def _intensity(self, now):
        if self.state == "active":
            return min(1.0, (now - self.t0) / RAMP_S)
        if self.state == "fading":
            return max(0.0, 1.0 - (now - self.t_end) / FADE_S)
        return 0.0

    def _frost_alpha(self, h, w):
        """Frosted-edge mask, strongest at the corners; cached per panel size."""
        if self._frost_key != (h, w):
            ys, xs = np.mgrid[0:h, 0:w]
            d = np.hypot((xs / w - 0.5) / 0.5, (ys / h - 0.5) / 0.5) / math.sqrt(2)
            self._frost = (np.clip((d - 0.45) / 0.55, 0, 1) ** 1.5)[..., None].astype(np.float32)
            self._frost_key = (h, w)
        return self._frost

    def draw_overlay(self, img, game, now):
        k = self._intensity(now)
        if k <= 0:
            return
        h, w = img.shape[:2]
        s = w / 560
        # Blue tint over the whole arena, with frost creeping in from the edges.
        cv2.addWeighted(img, 1 - 0.22 * k, np.full_like(img, ICE_TINT), 0.22 * k, 0, img)
        a = self._frost_alpha(h, w) * (0.75 * k)
        img[:] = (img * (1 - a) + np.array(FROST, np.float32) * a).astype(np.uint8)
        # Snowfall: positions are a pure function of time, so there's no particle state to keep.
        for fx, fy, sp, ph, sz in self.flakes:
            x = (fx + 0.03 * math.sin(now * (1 + sp) + ph * 6.28)) % 1.0
            y = (fy + now * (0.12 + 0.25 * sp)) % 1.0
            cv2.circle(img, (int(x * w), int(y * h)), max(1, int((1 + 2 * sz) * s * k)), SNOW, -1, cv2.LINE_AA)
        # A rotating ice crystal around the (slowed) ball.
        world = game.ball_world(now)
        if world:
            sx, sy, kz = proj(*world)
            c, r = (int(sx * w), int(sy * h)), max(3, int(BALL_R * w * kz))
            cv2.circle(img, c, r + max(3, r // 2), (255, 235, 180), 2, cv2.LINE_AA)
            for i in range(6):
                ang = now * 1.5 + i * math.pi / 3
                tip = (int(c[0] + math.cos(ang) * r * 2.4), int(c[1] + math.sin(ang) * r * 2.4))
                cv2.line(img, c, tip, (255, 245, 210), 1, cv2.LINE_AA)
