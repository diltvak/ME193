"""Per-leg streaming signal processing: gyro bias, gravity tracking, heading rate, swing (step) events.

Everything here is independent of how the sensor is strapped on:
  * steps come from swing peaks of the gyro projected on the leg's main rotation axis. That axis is
    found by PCA of the last few seconds of gyro (the knee-hinge direction), and its sign is chosen so
    the big swing peak is positive; the smaller opposite-signed heel-strike / push-off peaks are then
    ignored. Before the axis is known (first strides), |gyro| is used. One peak = one swing of this leg;
  * "up" is tracked in the sensor's own frame by a complementary filter (gyro propagation + accel
    correction), and the heading rate is the gyro projected on "up". Any mounting rotation leaves both
    unchanged, so the same code works upside down, sideways, on either leg, at any height on the shin.
  * gyro bias is re-estimated every time the leg is still, so it adapts to each sensor and to warm-up.

The same class runs live (samples arrive in packets) and offline (a whole file at once).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import config as C


class LegProcessor:
    def __init__(self, name: str):
        self.name = name
        self.bias = np.zeros(3)
        self.bias_ready = False            # True after the first still window
        self.up = None                     # unit "up" vector in the sensor frame
        self.last_t = None
        # still detection
        self._win_g = deque(maxlen=C.STILL_WIN)
        self._win_a = deque(maxlen=C.STILL_WIN)
        self._win_t = deque(maxlen=C.STILL_WIN)
        self._since_still_check = 0
        self.still_now = False
        # swing peak detection (3-point local max on smoothed |gyro|)
        self._mag_hist = deque(maxlen=5)
        self._prev2 = self._prev1 = 0.0
        self._prev1_t = None
        self._last_peak_t = -1e9
        self._peaks_hist = deque(maxlen=C.SWING_HISTORY)
        self._axis_buf = deque(maxlen=C.AXIS_WIN)    # recent bias-corrected gyro, for the swing axis
        self._axis = None
        self._since_axis = 0
        # outputs, one entry per processed sample
        self.t: list[float] = []
        self.yaw: list[float] = []         # heading rate about "up" (deg/s); + = counter-clockwise = left
        self.still: list[bool] = []
        self.swings: list[float] = []      # times of swing peaks (one per stride of this leg)

    # ------------------------------------------------------------------------------------
    def feed(self, t: np.ndarray, acc: np.ndarray, gyro: np.ndarray):
        """Process samples (t in s, acc in g (n,3), gyro in dps (n,3)). Older-than-seen samples are ignored."""
        for i in range(len(t)):
            ti = float(t[i])
            if self.last_t is not None and ti <= self.last_t:
                continue
            self._sample(ti, acc[i], gyro[i])

    def _sample(self, t, a, g):
        gap = self.last_t is None or t - self.last_t > C.BRIDGE_DT
        dt = 1.0 / C.FS if gap else t - self.last_t
        self.last_t = t
        an = float(np.sqrt(a @ a))

        if gap:                                        # orientation unknown after a gap: restart from accel
            self._win_g.clear(); self._win_a.clear(); self._win_t.clear()
            self._mag_hist.clear()
            self._prev2 = self._prev1 = 0.0
            self._prev1_t = None
            if an > 0.5:
                self.up = a / an

        # still detection and bias update (checked every 10 samples)
        self._win_g.append(g); self._win_a.append(an); self._win_t.append(t)
        self._since_still_check += 1
        if self._since_still_check >= 10 and len(self._win_g) == C.STILL_WIN:
            self._since_still_check = 0
            wg = np.asarray(self._win_g)
            still = (wg.std(axis=0).max() < C.STILL_GYRO_STD
                     and abs(np.mean(self._win_a) - 1.0) < C.STILL_ACC_TOL)
            self.still_now = bool(still)
            if still:
                m = wg.mean(axis=0)
                self.bias = m if not self.bias_ready else (1 - C.BIAS_ALPHA) * self.bias + C.BIAS_ALPHA * m
                self.bias_ready = True
        if self.still_now and self._since_still_check == 0 and an > 0.5:
            self.up = a / an                           # standing still: accel is exactly "up"

        w_dps = g - self.bias
        w = np.radians(w_dps)

        # gravity ("up") tracking: dv/dt = -w x v, nudged toward the accel direction when |a| ~ 1 g
        v = self.up if self.up is not None else (a / an if an > 0.5 else np.array([0.0, 1.0, 0.0]))
        v = v - np.cross(w, v) * dt
        if C.GRAVITY_ACC_BAND[0] < an < C.GRAVITY_ACC_BAND[1]:
            v = v + C.GRAVITY_GAIN * (a / an - v)
        v = v / np.sqrt(v @ v)
        self.up = v
        yaw = float(w_dps @ v)

        # swing peaks on the gyro projected on the swing axis (or |gyro| until that axis is known)
        self._axis_buf.append(w_dps)
        self._since_axis += 1
        if self._since_axis >= 50 and len(self._axis_buf) >= C.AXIS_WIN // 2:
            self._since_axis = 0
            self._update_axis()
        x = float(w_dps @ self._axis) if self._axis is not None else float(np.sqrt(w_dps @ w_dps))
        self._mag_hist.append(x)
        mag = sum(self._mag_hist) / len(self._mag_hist)
        if self._prev1_t is not None and self._prev1 >= self._prev2 and self._prev1 > mag:
            typical = np.median(self._peaks_hist) if len(self._peaks_hist) >= 3 else 0.0
            thr = max(C.SWING_MIN_DPS, C.SWING_REL * typical)
            if self._prev1 > thr and self._prev1_t - self._last_peak_t > C.SWING_REFRACTORY:
                self.swings.append(self._prev1_t)
                self._last_peak_t = self._prev1_t
                self._peaks_hist.append(self._prev1)
        self._prev2, self._prev1, self._prev1_t = self._prev1, mag, t

        self.t.append(t)
        self.yaw.append(yaw)
        self.still.append(self.still_now)

    def _update_axis(self):
        G = np.asarray(self._axis_buf)
        if G.std(axis=0).max() < C.AXIS_MIN_STD:      # not walking: keep the previous axis
            return
        evals, evecs = np.linalg.eigh(np.cov(G.T))
        p = evecs[:, -1]
        x = G @ p
        if -np.percentile(x, 1) > np.percentile(x, 99):   # swing peak is the larger extreme: make it positive
            p = -p
        if self._axis is None or self._axis @ p < 0.5:
            self._peaks_hist.clear()                  # peak heights change scale when the signal changes
        self._axis = p
