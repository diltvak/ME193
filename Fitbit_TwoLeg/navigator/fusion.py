"""Combine both legs into fixed-length ticks: steps taken, heading change, and turn events.

Each tick (config.TICK seconds) reports
  steps   - steps taken (both legs' swings), or None if neither leg had data (a Wi-Fi gap);
            with only one leg present, that leg's swings x 2 (one swing per stride = two steps)
  dhead   - heading change in degrees (+ = left), averaged over the legs present, or None
Turn events come from the heading change: a turn starts when the 1 s heading rate exceeds a
threshold and ends when it settles; walking-induced drift (1-2 deg/s) stays well below it.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from . import config as C
from .signals import LegProcessor


@dataclass
class Tick:
    t0: float
    t1: float
    steps: float | None
    dhead: float | None
    legs: tuple


@dataclass
class Turn:
    t: float           # time the turn finished
    t_start: float
    angle: float       # deg, + = left, - = right
    gap: bool          # a data gap happened during the turn (angle less reliable)
    t_detect: float = 0.0   # end of the tick in which the turn was recognised (after it settled)


class Fusion:
    def __init__(self):
        self.legs = {n: LegProcessor(n) for n in C.LEGS}
        self.t_next = None              # start of the next tick to produce
        self.ticks: list[Tick] = []
        self.turns: list[Turn] = []
        self._H = [0.0]                 # cumulative heading at the end of each tick
        self._turn_start = None
        self._calm = 0
        self._turn_gap = False

    def feed(self, leg, t, acc, gyro):
        self.legs[leg].feed(t, acc, gyro)

    def start(self, t):
        """Begin producing ticks at time t (the moment the walk starts)."""
        self.t_next = t
        self.ticks.clear(); self.turns.clear()
        self._H = [0.0]; self._turn_start = None; self._calm = 0

    def calibrated(self):
        """Which legs have a gyro bias estimate (seen standing still)."""
        return {n: lp.bias_ready for n, lp in self.legs.items()}

    def update(self, now):
        """Produce every tick that ends before now - LATENCY. Returns the new ticks."""
        new = []
        if self.t_next is None:
            return new
        while self.t_next + C.TICK <= now - C.LATENCY:
            tk = self._make_tick(self.t_next, self.t_next + C.TICK)
            self.t_next += C.TICK
            self.ticks.append(tk)
            self._turn_detect(tk)
            new.append(tk)
        return new

    # ------------------------------------------------------------------------------------
    def _leg_slice(self, lp: LegProcessor, t0, t1):
        i0, i1 = bisect_left(lp.t, t0), bisect_left(lp.t, t1)
        return i0, i1

    def _make_tick(self, t0, t1):
        present, steps, dheads = [], {}, []
        for name, lp in self.legs.items():
            i0, i1 = self._leg_slice(lp, t0, t1)
            if (i1 - i0) < C.COVER_MIN * C.TICK * C.FS:
                continue
            present.append(name)
            s0, s1 = bisect_left(lp.swings, t0), bisect_left(lp.swings, t1)
            steps[name] = s1 - s0
            tt = np.asarray(lp.t[max(i0 - 1, 0):i1])
            yy = np.asarray(lp.yaw[max(i0 - 1, 0):i1])
            dt = np.diff(tt, prepend=tt[0])
            dt[dt > C.BRIDGE_DT] = 0                                # long gap: rotation in it is unknown
            if i0 == 0:
                dt[0] = 0
            dheads.append(float((yy * dt)[1 if i0 > 0 else 0:].sum()))
        if not present:
            return Tick(t0, t1, None, None, ())
        n = sum(steps.values()) * (2 if len(present) == 1 else 1)
        return Tick(t0, t1, float(n), float(np.mean(dheads)), tuple(present))

    def _turn_detect(self, tk: Tick):
        self._H.append(self._H[-1] + (tk.dhead or 0.0))
        if tk.dhead is None:
            self._turn_gap = self._turn_gap or self._turn_start is not None
        k = len(self._H) - 1
        w = max(1, int(round(1.0 / C.TICK)))                       # 1 s rate
        if k < w:
            return
        rate = (self._H[k] - self._H[k - w]) / (w * C.TICK)
        if self._turn_start is None:
            if abs(rate) > C.TURN_RATE_END * 1.7:
                self._turn_start = k - w
                self._calm = 0
                self._turn_gap = any(t.dhead is None for t in self.ticks[-w:])
            return
        if abs(rate) < C.TURN_RATE_END:
            self._calm += 1
        else:
            self._calm = 0
        long = (k - self._turn_start) * C.TICK > 2 * C.TURN_WINDOW
        if self._calm * C.TICK >= C.TURN_SETTLE or long:
            k_end = k - self._calm + w // 2 if not long else k
            k_end = min(max(k_end, self._turn_start + 1), k)
            angle = self._H[k_end] - self._H[self._turn_start]
            if abs(angle) >= C.TURN_MIN_DEG:
                t_end = self.ticks[k_end - 1].t1
                t_start = self.ticks[self._turn_start].t0 if self._turn_start < len(self.ticks) else t_end
                self.turns.append(Turn(t_end, t_start, angle, self._turn_gap, tk.t1))
            self._turn_start = None
            self._turn_gap = False
