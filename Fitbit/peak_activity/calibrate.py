"""Fit the three thresholds from your labeled CSVs and report how well they hold up.

    python -m peak_activity.calibrate                 # use the axis in config.py (x)
    python -m peak_activity.calibrate --axis auto     # pick whichever axis separates the classes best
    python -m peak_activity.calibrate --axis y

Validation is leave-one-block-out: each recording is cut into --blocks contiguous time blocks, and
each block is classified with thresholds fitted on the *other* blocks. (With one file per activity
this is the best available stand-in for leave-one-session-out; it is optimistic.)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from activity.data_io import discover_files, load_csv, resample_uniform, sliding_windows

from .config import ACTIVITIES, AXES, DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, PeakConfig
from .peak import classify_peak, peak_accel


def collect_windows(data_dir, cfg: PeakConfig, blocks: int):
    """-> list of (label, block_id, window (n,3)) over every recording found."""
    found = discover_files(data_dir)
    out = []
    for label, paths in found.items():
        if label not in ACTIVITIES:
            continue
        for fi, p in enumerate(paths):
            t, X = load_csv(p)
            _, Xu, valid = resample_uniform(t, X, cfg)
            wins = [(s, w[:, :3]) for s, w in sliding_windows(Xu, valid, cfg)]
            for k, (s, w) in enumerate(wins):
                out.append((label, (label, fi, min(blocks - 1, k * blocks // len(wins))), w))
    return out


def peaks_by_class(windows, axis, pct, keep=None):
    d = {a: [] for a in ACTIVITIES}
    for label, block, w in windows:
        if keep is None or keep(block):
            d[label].append(peak_accel(w[:, axis], pct))
    return {k: np.array(v) for k, v in d.items()}


def fit_threshold(lo: np.ndarray, hi: np.ndarray) -> float:
    """Threshold between two classes minimising the mean per-class error rate.

    If several thresholds tie (e.g. the classes are fully separable), the middle of the tied
    range is returned, which maximises the margin on both sides.
    """
    vals = np.sort(np.concatenate([lo, hi]))
    cand = (vals[:-1] + vals[1:]) / 2
    err = np.array([((lo >= c).mean() + (hi < c).mean()) / 2 for c in cand])
    best = cand[err <= err.min() + 1e-12]
    return float(np.median(best))


def fit_thresholds(peaks):
    order = ["standing", "biking", "walking", "running"]
    return [fit_threshold(peaks[a], peaks[b]) for a, b in zip(order, order[1:])]


def axis_report(windows, pct):
    """Per axis: the smallest ratio between neighbouring classes' (worst-case) peaks. >1 = no overlap."""
    order = ["standing", "biking", "walking", "running"]
    rows = []
    for ax in range(3):
        pk = peaks_by_class(windows, ax, pct)
        ratios = [pk[b].min() / max(pk[a].max(), 1e-3) for a, b in zip(order, order[1:])]
        rows.append((ax, ratios))
    return rows


def confusion(true, pred):
    m = np.zeros((len(ACTIVITIES),) * 2, int)
    for t, p in zip(true, pred):
        m[ACTIVITIES.index(t), ACTIVITIES.index(p)] += 1
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--axis", default=None, help="x | y | z | auto (default: config.py)")
    ap.add_argument("--blocks", type=int, default=4, help="time blocks per recording for validation")
    args = ap.parse_args(argv)

    cfg = PeakConfig()
    windows = collect_windows(args.data_dir, cfg, args.blocks)
    have = {w[0] for w in windows}
    if have != set(ACTIVITIES):
        sys.exit(f"need recordings for all of {ACTIVITIES}; found {sorted(have)} in {args.data_dir}")
    print("windows per class:", {a: sum(w[0] == a for w in windows) for a in ACTIVITIES})

    print(f"\nAxis comparison (p{cfg.peak_pct:.0f} of |a - median|; each ratio = lowest peak of the class "
          f"above / highest of the class below; >1 means no overlap):")
    print(f"  {'axis':<5}{'bike/stand':>12}{'walk/bike':>12}{'run/walk':>12}{'worst':>9}")
    rep = axis_report(windows, cfg.peak_pct)
    for ax, r in rep:
        print(f"  {AXES[ax]:<5}" + "".join(f"{x:>12.2f}" for x in r) + f"{min(r):>9.2f}")
    best = max(rep, key=lambda r: min(r[1]))[0]

    if args.axis is None:
        axis = cfg.axis
    elif args.axis == "auto":
        axis = best
    else:
        axis = AXES.index(args.axis.lower())
    cfg = replace(cfg, axis=axis)
    print(f"\nusing axis {cfg.axis_name}" + ("" if axis == best else f"  (note: axis {AXES[best]} separates best)"))

    # --- leave-one-block-out validation ---
    blocks = sorted({w[1] for w in windows})
    true, pred = [], []
    for held in blocks:
        pk = peaks_by_class(windows, axis, cfg.peak_pct, keep=lambda b: b != held)
        a, b, c = fit_thresholds(pk)
        fold = replace(cfg, t_stand_bike=a, t_bike_walk=b, t_walk_run=c)
        for label, block, w in windows:
            if block == held:
                true.append(label)
                pred.append(classify_peak(peak_accel(w[:, axis], cfg.peak_pct), fold))
    m = confusion(true, pred)
    print(f"\nLeave-one-block-out accuracy: {np.trace(m) / m.sum():.1%}  "
          f"(balanced: {np.mean(np.diag(m) / m.sum(1)):.1%})")
    print("confusion (rows = true, cols = predicted):")
    print(f"  {'':<10}" + "".join(f"{a[:7]:>9}" for a in ACTIVITIES))
    for a, row in zip(ACTIVITIES, m):
        print(f"  {a:<10}" + "".join(f"{v:>9}" for v in row))

    # --- final fit on everything ---
    pk = peaks_by_class(windows, axis, cfg.peak_pct)
    print("\nPeak (g) per class: min / median / max")
    for a in ACTIVITIES:
        print(f"  {a:<9}{pk[a].min():6.2f} /{np.median(pk[a]):6.2f} /{pk[a].max():6.2f}")
    a, b, c = fit_thresholds(pk)
    cfg = replace(cfg, t_stand_bike=round(a, 3), t_bike_walk=round(b, 3), t_walk_run=round(c, 3))
    print(f"\nthresholds: standing<{cfg.t_stand_bike}  biking<{cfg.t_bike_walk}  "
          f"walking<{cfg.t_walk_run}  running>=  (g, axis {cfg.axis_name})")
    if pk["walking"].min() - pk["biking"].max() < 0.2:
        print("  warning: biking and walking peaks are close - record more of both to firm up that threshold")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(args.model_dir / "thresholds.json")
    print(f"saved {args.model_dir / 'thresholds.json'}")


if __name__ == "__main__":
    main()
