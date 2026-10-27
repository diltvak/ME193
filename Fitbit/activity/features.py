"""Per-window feature extraction.

`extract_features` is the one place to add/remove features: it returns an ordered dict of
name -> float, and the trainer/classifier pick the feature list up from it automatically
(retrain after changing it). All features are built from magnitudes or sorted/PCA-projected
quantities so they do not depend on how the sensor is strapped on.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from .config import Config

EPS = 1e-9


def _spectral_periodicity(x, cfg: Config):
    """FFT-based periodicity of a 1-D signal.

    Returns (f0, fund_frac, harm_ratio):
      f0          dominant frequency in [f_min, f_max] Hz
      fund_frac   power within +/- fund_halfband of f0 / total power (0.3..f_total_max Hz);
                  ~1 for a pure sinusoid, low for sharp/impulsive waveforms
      harm_ratio  power at 2f0..4f0 / power at f0 (large for sharp waveforms)
    """
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), cfg.nfft)) ** 2
    f = np.fft.rfftfreq(cfg.nfft, 1.0 / cfg.fs)
    band = (f >= cfg.f_min_hz) & (f <= cfg.f_max_hz)
    total = spec[(f >= 0.3) & (f <= cfg.f_total_max_hz)].sum() + EPS
    f0 = f[band][np.argmax(spec[band])]
    hb = cfg.fund_halfband_hz
    fund = spec[np.abs(f - f0) <= hb].sum()
    harm = sum(spec[np.abs(f - m * f0) <= hb].sum() for m in (2, 3, 4) if m * f0 < cfg.fs / 2)
    return f0, min(1.0, fund / total), harm / (fund + EPS)


def _acf_peak(x, cfg: Config):
    """Highest normalised autocorrelation at lags matching f_min..f_max (1 = perfectly periodic)."""
    x = x - x.mean()
    denom = np.dot(x, x) + EPS
    lo, hi = int(cfg.fs / cfg.f_max_hz), min(int(cfg.fs / cfg.f_min_hz), len(x) - 2)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:] / denom
    return float(ac[lo:hi + 1].max())


def _principal_axis(gyro):
    """Project 3-axis gyro onto its first principal component (the main swing axis)."""
    g = gyro - gyro.mean(axis=0)
    _, _, vt = np.linalg.svd(g, full_matrices=False)
    return g @ vt[0]


def extract_features(win, cfg: Config) -> dict:
    """Compute the feature dict for one window.

    win: (window_n, >=6) array with columns ax ay az gx gy gz (g, dps), uniformly sampled.
    """
    acc, gyro = win[:, :3], win[:, 3:6]
    a = np.linalg.norm(acc, axis=1)           # |acc|, includes gravity (~1 g at rest)
    g = np.linalg.norm(gyro, axis=1)
    g1 = _principal_axis(gyro)                # dominant swing axis (orientation independent)
    fs = cfg.fs
    F = {}

    # --- intensity: acceleration magnitude
    F["a_mean"] = a.mean()
    F["a_std"] = a.std()
    F["a_range"] = np.ptp(a)
    F["a_jerk_rms"] = np.sqrt(np.mean((np.diff(a) * fs) ** 2))

    # --- intensity: gyro magnitude
    F["g_mean"] = g.mean()
    F["g_std"] = g.std()
    F["g_range"] = np.ptp(g)

    # --- per-axis spread, sorted so sensor axis labelling does not matter
    for name, arr in (("acc", acc), ("gyro", gyro)):
        s = np.sort(arr.std(axis=0))[::-1]
        F[f"{name}_axstd_hi"], F[f"{name}_axstd_mid"], F[f"{name}_axstd_lo"] = s
        F[f"{name}_axis_dominance"] = s[0] / (s.sum() + EPS)

    # --- impact
    F["a_peak"] = a.max()
    peaks, _ = find_peaks(a, prominence=cfg.peak_prominence_g,
                          distance=max(1, int(cfg.peak_min_dist_s * fs)))
    F["a_n_peaks"] = len(peaks)
    F["a_kurtosis"] = kurtosis(a)

    # --- periodicity of |acc|
    F["a_f0"], F["a_fund_frac"], F["a_harm_ratio"] = _spectral_periodicity(a, cfg)
    F["a_acf_peak"] = _acf_peak(a, cfg)

    # --- gyro shape along the main swing axis
    F["g1_std"] = g1.std()
    F["g1_f0"], F["g1_fund_frac"], F["g1_harm_ratio"] = _spectral_periodicity(g1, cfg)
    F["g1_acf_peak"] = _acf_peak(g1, cfg)
    F["g1_kurtosis"] = kurtosis(g1)           # sine = -1.5; sharp swing/stance profiles are higher
    F["g1_skew_abs"] = abs(skew(g1))
    F["g1_crest"] = np.abs(g1).max() / (np.sqrt(np.mean(g1 ** 2)) + EPS)   # sine = 1.41

    return {k: float(v) for k, v in F.items()}


def feature_names(cfg: Config):
    """Ordered feature names (derived from extract_features so they never drift)."""
    rng = np.random.default_rng(0)
    return list(extract_features(rng.normal(size=(cfg.window_n, 6)), cfg))


def to_vector(feats: dict, names):
    return np.array([feats[n] for n in names], dtype=float)
