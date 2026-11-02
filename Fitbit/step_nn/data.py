"""Loading, trimming, stride pseudo-labels and feature extraction.

A "record" is a dict:
    name   file stem
    t, X   uniform-rate time (s) and the six IMU channels (ax ay az g, gx gy gz dps)
    gap    bool per sample, True where the raw stream had a dropped-packet gap
    raw_t, raw_X7   the untouched samples (for replaying through the streaming counter)
    peaks  stride peak indices into t/X (gz maxima, one per stride of the sensor leg)
    L      stride length in metres if the distance is known, else None
    n_filled   strides inserted into packet-loss gaps by the labeler
    sl     slice of the walking bout (trimmed ends, with padding)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from activity.data_io import load_csv, resample_uniform

from .config import (EXTRA_WALK_FILES, N_TRIALS, NEGATIVE_FILES, TRIAL_DISTANCE_M, TRUE_STEPS,
                     StepConfig)

ACC_SCALE = 4.0      # sensor range, g
GYRO_SCALE = 500.0   # sensor range, dps
N_FEATURES = 8


# ----------------------------------------------------------------------------
# loading + trimming
# ----------------------------------------------------------------------------
def to_uniform(t, X7, cfg: StepConfig):
    """Irregular samples -> (grid, X (n,6), gap (n,)). gap marks dropped-packet stretches."""
    t = np.maximum.accumulate(np.asarray(t, float))
    grid, Xu, _ = resample_uniform(t, X7, cfg)
    hi = np.clip(np.searchsorted(t, grid, side="right"), 1, len(t) - 1)
    gap = (t[hi] - t[hi - 1]) > cfg.gap_flag_s
    return grid, Xu[:, :6], gap


def trim_ends(X, cfg: StepConfig):
    """Slice of the walking bout: first to last moment the gyro is active, plus padding."""
    gm = np.linalg.norm(X[:, 3:6], axis=1)
    k = max(1, int(cfg.trim_win_s * cfg.fs))
    box = np.ones(k) / k
    mean = np.convolve(gm, box, "same")
    std = np.sqrt(np.maximum(np.convolve(gm ** 2, box, "same") - mean ** 2, 0.0))
    active = np.flatnonzero(std > cfg.trim_gyro_std_dps)
    if len(active) == 0:
        return slice(0, len(X))
    pad = int(cfg.trim_pad_s * cfg.fs)
    return slice(max(0, active[0] - pad), min(len(X), active[-1] + pad + 1))


# ----------------------------------------------------------------------------
# stride pseudo-labels
# ----------------------------------------------------------------------------
def stride_peaks(X, gap, cfg: StepConfig):
    """Classical stride detector, used only to make training labels.

    The ankle's gz (swing about the shank axis) has one large positive lobe per stride of the
    instrumented leg. Steps: low-pass -> autocorrelation for the stride period -> prominence
    peak picking -> merge peaks closer than ~half a period -> fill strides hidden inside
    dropped-packet gaps. Returns (peak indices, number filled, period in samples).
    """
    fs = cfg.fs
    gz = sosfiltfilt(butter(2, cfg.label_lp_hz, fs=fs, output="sos"), X[:, 5])
    gz = gz - np.median(gz)
    amp = np.percentile(gz, 95)
    if amp < cfg.label_min_prom_dps:
        return np.empty(0, int), 0, 0
    ac = np.correlate(gz, gz, "full")[len(gz) - 1:]
    lo, hi = int(0.5 * fs), min(int(2.0 * fs), len(ac) - 1)
    period = lo + int(np.argmax(ac[lo:hi]))
    prom = max(cfg.label_prom_frac * amp, cfg.label_min_prom_dps)
    p, _ = find_peaks(gz, prominence=prom, distance=int(0.6 * period))
    if len(p) < 3:
        return p, 0, period
    med = float(np.median(np.diff(p)))
    p = list(p)
    merged = True
    while merged:                                           # a double bump is one stride
        merged = False
        for i in range(len(p) - 1):
            if p[i + 1] - p[i] < cfg.label_merge_frac * med:
                p.pop(i + 1 if gz[p[i + 1]] < gz[p[i]] else i)
                merged = True
                break
    filled, out = 0, [p[0]]
    for a, b in zip(p[:-1], p[1:]):                         # only where data really was missing
        if b - a > cfg.label_fill_frac * med and gap[a:b].any():
            n_missing = max(1, int(round((b - a) / med)) - 1)
            for j in range(1, n_missing + 1):
                out.append(int(a + (b - a) * j / (n_missing + 1)))
                filled += 1
        out.append(b)
    return np.array(out, int), filled, period


# ----------------------------------------------------------------------------
# network input
# ----------------------------------------------------------------------------
def make_features(X):
    """(T,6) raw g/dps -> (T,8) roughly in [-1, 1]: scaled axes plus |a|-1 and |gyro| magnitudes."""
    acc, gyro = X[:, :3] / ACC_SCALE, X[:, 3:6] / GYRO_SCALE
    amag = (np.linalg.norm(X[:, :3], axis=1) - 1.0) / 3.0
    gmag = np.linalg.norm(X[:, 3:6], axis=1) / (GYRO_SCALE * 1.5)
    return np.column_stack([acc, gyro, amag, gmag]).astype(np.float32)


def gaussian_target(peaks, n, sigma):
    """Soft 'a stride peak is here' target."""
    out = np.zeros(n, np.float32)
    if len(peaks):
        idx = np.arange(n)[:, None]
        out = np.exp(-0.5 * ((idx - np.asarray(peaks)[None, :]) / sigma) ** 2).max(axis=1).astype(np.float32)
    return out


def stride_features(X, peak, prev_peak, cfg: StepConfig):
    """Hand-built features for one stride, used by the ridge-regression baseline.

    [cadence (Hz), peak smoothed gz (dps), gyro peak-to-peak (dps), 95th pct of ||a|-1| (g)]
    over a +/-0.4 s window around the stride peak.
    """
    fs = cfg.fs
    w = int(0.4 * fs)
    a, b = max(0, peak - w), min(len(X), peak + w + 1)
    seg = X[a:b]
    period = (peak - prev_peak) / fs if prev_peak is not None else cfg.default_period_s
    period = float(np.clip(period, 0.4, 2.0))
    gz = seg[:, 5]
    amag = np.abs(np.linalg.norm(seg[:, :3], axis=1) - 1.0)
    return np.array([1.0 / period, gz.max(), np.ptp(gz), np.percentile(amag, 95)])


# ----------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------
def load_record(path, cfg: StepConfig, label=True, distance_m=None):
    path = Path(path)
    raw_t, raw_X7 = load_csv(path)
    t, X, gap = to_uniform(raw_t, raw_X7, cfg)
    rec = {"name": path.stem, "t": t, "X": X, "gap": gap, "raw_t": raw_t, "raw_X7": raw_X7,
           "peaks": np.empty(0, int), "n_filled": 0, "L": None, "sl": trim_ends(X, cfg)}
    if label:
        rec["peaks"], rec["n_filled"], _ = stride_peaks(X, gap, cfg)
        if distance_m is not None and len(rec["peaks"]):
            rec["L"] = distance_m / len(rec["peaks"])        # equal-length strides within a trial
    return rec


def load_all(data_dir, cfg: StepConfig):
    """Returns dict of record lists: trials (known distance), counts (known step count),
    extra (pseudo-labelled walking), negatives (no walking)."""
    d = Path(data_dir)
    out = {"trials": [], "counts": [], "extra": [], "negatives": []}
    for i in range(1, N_TRIALS + 1):
        out["trials"].append(load_record(d / f"trial_{i}.csv", cfg, distance_m=TRIAL_DISTANCE_M))
    for name, n in TRUE_STEPS.items():
        rec = load_record(d / f"{name}.csv", cfg)
        rec["true_steps"] = n
        out["counts"].append(rec)
    for name in EXTRA_WALK_FILES:
        if (d / f"{name}.csv").exists():
            out["extra"].append(load_record(d / f"{name}.csv", cfg))
    for name in NEGATIVE_FILES:
        if (d / f"{name}.csv").exists():
            out["negatives"].append(load_record(d / f"{name}.csv", cfg, label=False))
    return out
