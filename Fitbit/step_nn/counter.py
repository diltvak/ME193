"""Streaming step + distance counter, shared by live.py and evaluate.py.

Raw samples go in (any chunking, irregular timestamps, gaps allowed); every `hop_s` the last
`infer_window_s` is resampled to 100 Hz and run through StepNet. A peak of the step probability is
*final* once `lookahead_s` of later data exists (the net's receptive field), so each stride is
counted exactly once. Each stride adds its predicted length to the distance.

Counting convention: the sensor is on one ankle, so one detected peak is one stride = 2 steps.
Your three hand-counted walks (33/29/23 steps) gave 17/15/12 peaks, i.e. steps = 2*strides - 1:
the sensor leg took the first and last step. `first_step_on_sensor_leg` selects that (default) or
the plain 2*strides.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from activity.data_io import resample_uniform

from .config import StepConfig
from .data import stride_features
from .model import predict


class RidgeLength:
    """Baseline stride-length model: ridge regression on four hand-built per-stride features."""

    def __init__(self, mean, std, w, b):
        self.mean, self.std, self.w, self.b = mean, std, w, b

    @classmethod
    def fit(cls, records, cfg: StepConfig):
        F, y = [], []
        for rec in records:
            if rec["L"] is None:
                continue
            p = rec["peaks"]
            for k, pk in enumerate(p):
                F.append(stride_features(rec["X"], pk, p[k - 1] if k else None, cfg))
                y.append(rec["L"])
        F, y = np.array(F), np.array(y)
        mean, std = F.mean(0), F.std(0) + 1e-9
        Z = (F - mean) / std
        w = np.linalg.solve(Z.T @ Z + cfg.ridge_alpha * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
        return cls(mean, std, w, float(y.mean()))

    def __call__(self, feats):
        return float(((feats - self.mean) / self.std) @ self.w + self.b)

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, w=self.w, b=self.b)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mean"], z["std"], z["w"], float(z["b"]))


class StepCounter:
    def __init__(self, cfg: StepConfig, net, ridge: RidgeLength | None = None):
        self.cfg, self.net, self.ridge = cfg, net, ridge
        self.reset()

    def reset(self):
        self.t = np.empty(0)
        self.X = np.empty((0, 7))
        self.events = []            # dicts: t, length (used), nn_length, ridge_length
        self.final_until = -np.inf  # peaks at or before this time have been decided
        self.next_eval = None
        self.last_prob = (np.empty(0), np.empty(0), np.empty(0))   # (t, probability, X) of last window
        self._prev_peak_t = None
        self.maybe = []             # final but sub-threshold peaks (0.2 <= p < peak_thresh): possible missed strides

    # ---- results --------------------------------------------------------------
    @property
    def strides(self):
        return len(self.events)

    @property
    def steps(self):
        if self.strides == 0:
            return 0
        return 2 * self.strides - (1 if self.cfg.first_step_on_sensor_leg else 0)

    @property
    def weak_strides(self):
        """Counted strides the net was not confident about (peak probability below weak_conf)."""
        return sum(e["conf"] < self.cfg.weak_conf for e in self.events)

    @property
    def steps_range(self):
        """(low, high) steps if every weak stride were false / every sub-threshold candidate were real."""
        def steps(k):
            return 0 if k <= 0 else 2 * k - (1 if self.cfg.first_step_on_sensor_leg else 0)
        return steps(self.strides - self.weak_strides), steps(self.strides + len(self.maybe))

    @property
    def distance_m(self):
        return float(sum(e["length"] for e in self.events))

    @property
    def cadence_spm(self):
        """Steps per minute over the last few strides (0 if unknown)."""
        if len(self.events) < 3:
            return 0.0
        ts = [e["t"] for e in self.events[-5:]]
        return 120.0 * (len(ts) - 1) / (ts[-1] - ts[0])

    # ---- streaming ------------------------------------------------------------
    def update(self, t, X7):
        """Feed raw samples (t seconds, X7 = ax ay az gx gy gz temp). Returns new events."""
        cfg = self.cfg
        t = np.asarray(t, float)
        if len(t) == 0:
            return []
        if len(self.t) and t[0] - self.t[-1] > cfg.gap_reset_s:     # long outage: start a fresh window
            self.t, self.X = np.empty(0), np.empty((0, 7))
            self.final_until = t[0]
            self.next_eval = None
        self.t = np.concatenate([self.t, t])
        self.X = np.vstack([self.X, X7])
        keep = self.t > self.t[-1] - cfg.buffer_s
        self.t, self.X = self.t[keep], self.X[keep]
        if self.next_eval is None:
            self.next_eval = self.t[0] + cfg.infer_window_s * 0.5
        if self.t[-1] < self.next_eval:
            return []
        self.next_eval = self.t[-1] + cfg.hop_s
        return self._run(cfg.lookahead_s)

    def flush(self):
        """End of stream: decide the remaining peaks without waiting for look-ahead data."""
        return self._run(0.0) if len(self.t) > 2 else []

    def _run(self, lookahead):
        cfg = self.cfg
        recent = self.t >= self.t[-1] - cfg.infer_window_s - 0.1
        tu, Xu, _ = resample_uniform(self.t[recent], self.X[recent], cfg)
        if len(Xu) < int(1.5 * cfg.fs):
            return []
        Xu = Xu[:, :6]
        prob, length = predict(self.net, Xu)
        self.last_prob = (tu, prob, Xu)
        limit = tu[-1] - lookahead
        idx, _ = find_peaks(prob, height=cfg.peak_thresh, distance=int(cfg.refractory_s * cfg.fs))
        new = []
        counted = {e["t"] for e in self.events}
        seen = {m["t"] for m in self.maybe}
        weak_idx, _ = find_peaks(prob, height=cfg.maybe_thresh, distance=int(cfg.refractory_s * cfg.fs))
        for i in weak_idx:
            tp = float(tu[i])
            if prob[i] >= cfg.peak_thresh or tp <= self.final_until or tp > limit or tp in seen:
                continue
            near = [c for c in list(counted) + [self._prev_peak_t or -1e9] if abs(c - tp) < cfg.refractory_s]
            if not near and not any(abs(m["t"] - tp) < cfg.refractory_s for m in self.maybe):
                self.maybe.append({"t": tp, "conf": float(prob[i])})
        for i in idx:
            tp = tu[i]
            if tp <= self.final_until or tp > limit:
                continue
            if self._prev_peak_t is not None and tp - self._prev_peak_t < cfg.refractory_s:
                continue
            nn_len = float(length[max(0, i - 3):i + 4].mean())
            r_len = None
            if self.ridge is not None:
                prev = None
                if self._prev_peak_t is not None:
                    prev = i - int(round((tp - self._prev_peak_t) * cfg.fs))
                    prev = prev if prev >= 0 else None
                r_len = self.ridge(stride_features(Xu, int(i), prev, cfg))
            use = r_len if (cfg.len_source == "ridge" and r_len is not None) else nn_len
            ev = {"t": float(tp), "conf": float(prob[i]), "length": use, "nn_length": nn_len, "ridge_length": r_len}
            self.events.append(ev)
            new.append(ev)
            self._prev_peak_t = tp
        self.final_until = max(self.final_until, limit)
        return new


def run_recording(counter: StepCounter, raw_t, raw_X7, chunk=25):
    """Replay a recorded file through a counter as fast as possible, in 0.25 s chunks."""
    counter.reset()
    for s in range(0, len(raw_t), chunk):
        counter.update(raw_t[s:s + chunk], raw_X7[s:s + chunk])
    counter.flush()
    return counter
