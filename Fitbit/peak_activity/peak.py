"""The whole algorithm: one number per window (peak acceleration on one axis) and three thresholds."""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from .config import PeakConfig


def peak_accel(a, pct: float = 95.0) -> float:
    """Peak acceleration (g) of one axis over a window.

    Gravity (and whatever the strap orientation does to it) is removed by subtracting the window
    median; the peak is then a high percentile of |deviation|. A percentile instead of the raw max
    keeps one noise spike or a clipped (+-4 g) sample from deciding the label on its own.
    """
    a = np.asarray(a, dtype=float)
    return float(np.percentile(np.abs(a - np.median(a)), pct))


def window_peak(win, cfg: PeakConfig) -> float:
    """Peak of cfg.axis for a (n, >=3) window of ax ay az ... columns."""
    return peak_accel(win[:, cfg.axis], cfg.peak_pct)


def classify_peak(peak: float, cfg: PeakConfig) -> str:
    if peak < cfg.t_stand_bike:
        return "standing"
    if peak < cfg.t_bike_walk:
        return "biking"
    if peak < cfg.t_walk_run:
        return "walking"
    return "running"


class MajorityVote:
    """Most common label over the last n windows (ties -> most recent)."""

    def __init__(self, n: int):
        self.buf = deque(maxlen=n)

    def update(self, label: str) -> str:
        self.buf.append(label)
        return self.current()

    def current(self):
        if not self.buf:
            return None
        counts = Counter(self.buf)
        best = max(counts.values())
        for lab in reversed(self.buf):
            if counts[lab] == best:
                return lab

    def reset(self, label=None):
        self.buf.clear()
        if label is not None:
            self.buf.extend([label] * self.buf.maxlen)
