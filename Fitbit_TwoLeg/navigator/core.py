"""Navigator: IMU samples in, walking directions out. Used by the live GUI, replay, and evaluation."""
from __future__ import annotations

import json

from . import config as C
from .fusion import Fusion
from .tracker import Guidance, RouteTracker


def load_route(path=C.ROUTE_FILE):
    with open(path) as f:
        return json.load(f)


class Navigator:
    def __init__(self, route: dict, user_height: float, seed=0):
        self.route = route
        self.user_height = user_height
        self.step_len0 = C.STEP_LENGTH_PER_HEIGHT * user_height
        self.fusion = Fusion()
        self.tracker = None
        self.seed = seed
        self.messages: list[tuple[float, str]] = []   # (t, one-off alerts: wrong turn, back on route)

    def feed(self, leg, t, acc, gyro):
        self.fusion.feed(leg, t, acc, gyro)

    def calibrated(self):
        return self.fusion.calibrated()

    def start(self, t):
        """The walker is at the start point and begins now."""
        self.fusion.start(t)
        self.tracker = RouteTracker(self.route, self.step_len0, seed=self.seed)
        self.messages.clear()

    @property
    def started(self):
        return self.tracker is not None

    def update(self, now) -> Guidance | None:
        if self.tracker is None:
            return None
        n_turns = len(self.fusion.turns)
        for tk in self.fusion.update(now):
            self.tracker.tick(tk)
            while n_turns < len(self.fusion.turns) and self.fusion.turns[n_turns].t_detect <= tk.t1:
                ev = self.fusion.turns[n_turns]
                n_turns += 1
                msg = self.tracker.turn(ev)
                if msg:
                    self.messages.append((ev.t, msg))
        return self.tracker.guidance()
