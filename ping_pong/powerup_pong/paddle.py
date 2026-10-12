"""Paddle position and angle from the pose landmarks."""

import math
from collections import deque

from pose_detection import PL
from swing import MIN_VISIBILITY

from powerup_pong.config import PADDLE_CENTER, PADDLE_GAIN, PADDLE_SMOOTHING


class Paddle:
    """Smoothed paddle position (panel fractions) and handle angle taken from the pose."""

    def __init__(self):
        self.pos = None
        self.dir = (0.0, 1.0)     # unit vector from the paddle toward the elbow (handle side)
        self.tracked = False
        self.trail = deque(maxlen=8)

    @property
    def angle(self):
        return math.atan2(self.dir[1], self.dir[0])

    @property
    def xy(self):
        return self.pos if self.tracked else None

    def update(self, landmarks, arm):
        wrist = landmarks[PL[f"{arm.upper()}_WRIST"].value] if landmarks else None
        if wrist is None or wrist.visibility < MIN_VISIBILITY:
            self.tracked = False
            return
        elbow = landmarks[PL[f"{arm.upper()}_ELBOW"].value]
        mx, my = 1 - wrist.x, wrist.y     # mirrored to match the flipped view
        target = tuple(min(0.97, max(0.03, 0.5 + (v - c) * PADDLE_GAIN))
                       for v, c in zip((mx, my), PADDLE_CENTER))
        if self.pos is None:
            self.pos = target
        else:
            self.pos = tuple(p + PADDLE_SMOOTHING * (t - p) for p, t in zip(self.pos, target))
        if elbow.visibility >= MIN_VISIBILITY:
            dx, dy = (wrist.x - elbow.x) * 4, (elbow.y - wrist.y) * 3   # panel is 4:3
            norm = math.hypot(dx, dy) or 1.0
            self.dir = tuple(d + 0.4 * (n / norm - d) for d, n in zip(self.dir, (dx / norm, dy / norm)))
        self.tracked = True
        self.trail.append(self.pos)
