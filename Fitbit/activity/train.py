"""Train and evaluate the hybrid classifier.

    python -m activity.train [--data-dir ..] [--out activity/models] [--blocks 4]

Stage 1 (standing) is a rule, so only the random forest is trained, on walking/running/biking
windows. Evaluation runs the whole hybrid pipeline on held-out sessions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from .classifier import is_standing, rule_based_moving
from .config import ACTIVITIES, DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, Config
from .data_io import discover_files, load_csv, resample_uniform, sliding_windows
from .features import extract_features, feature_names, to_vector


def make_forest(seed=0):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def build_dataset(files, cfg: Config, blocks: int):
    """Window every recording. Returns (feature dicts, labels, groups, source tags).

    Sessions: a "session" is one independent recording. If a class has >= 3 files, each file is
    one session. Otherwise (e.g. one file per activity, like now) each file is cut into `blocks`
    contiguous time blocks that act as pseudo-sessions. Windows never straddle a block boundary.
    Group g = unit index modulo the smallest number of units any class has, so every CV fold
    holds out "session g" of every activity.
    """
    units = defaultdict(list)          # label -> list of (name, t_u, X_u, valid)
    for label, paths in files.items():
        for p in paths:
            t, X = load_csv(p)
            _, Xu, valid = resample_uniform(t, X, cfg)
            if len(paths) >= 3:
                units[label].append((p.name, Xu, valid))
            else:
                edges = np.linspace(0, len(Xu), blocks + 1).astype(int)
                for b in range(blocks):
                    sl = slice(edges[b], edges[b + 1])
                    units[label].append((f"{p.name}#{b}", Xu[sl], valid[sl]))
    n_groups = min(len(u) for u in units.values())

    feats, labels, groups, tags = [], [], [], []
    for label, ulist in units.items():
        for idx, (name, Xu, valid) in enumerate(ulist):
            for _, w in sliding_windows(Xu, valid, cfg):
                feats.append(extract_features(w, cfg))
                labels.append(label)
                groups.append(idx % n_groups)
                tags.append(name)
    return feats, np.array(labels), np.array(groups), tags


def evaluate(F, y, groups, names, cfg: Config):
    """Leave-one-session-out CV of the full hybrid pipeline (+ the rule-tree baseline).

    WHY NOT RANDOM WINDOW SPLITS: with 50% overlap, neighbouring windows share half their
    samples, and even non-overlapping neighbours come from the same bout (same speed, same
    posture, same sensor tightness). A random split puts near-duplicates in train and test, so
    accuracy looks great but only measures memorisation. Holding out an entire session
    forces the model to generalise to data it has never seen from that recording.
    """
    y_true, y_hat, y_tree = [], [], []
    for g in np.unique(groups):
        tr = (groups != g) & (y != "standing")
        tr &= np.array([not is_standing(f, cfg) for f in F])      # drop "moving" windows that are still
        te = groups == g
        model = make_forest().fit(F_mat(F, tr, names), y[tr])
        for i in np.where(te)[0]:
            f = F[i]
            y_true.append(y[i])
            y_hat.append("standing" if is_standing(f, cfg)
                         else model.predict(to_vector(f, names).reshape(1, -1))[0])
            y_tree.append("standing" if is_standing(f, cfg) else rule_based_moving(f, cfg))
    return np.array(y_true), np.array(y_hat), np.array(y_tree)


def F_mat(F, mask, names):
    return np.array([to_vector(f, names) for f, m in zip(F, mask) if m])


def report(title, y_true, y_pred):
    print(f"\n=== {title} ===")
    cm = confusion_matrix(y_true, y_pred, labels=ACTIVITIES)
    w = 10
    print(" " * w + "".join(f"{a[:8]:>{w}}" for a in ACTIVITIES) + "   <- predicted")
    for a, row in zip(ACTIVITIES, cm):
        print(f"{a:<{w}}" + "".join(f"{v:>{w}}" for v in row))
    print(classification_report(y_true, y_pred, labels=ACTIVITIES, zero_division=0, digits=3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--blocks", type=int, default=4, help="pseudo-sessions per file when a class has <3 files")
    ap.add_argument("--config", type=Path, help="start from a saved config.json")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    files = discover_files(args.data_dir)
    missing = [a for a in ACTIVITIES if a not in files]
    if missing:
        raise SystemExit(f"No labeled files for {missing} in {args.data_dir}. "
                         "Run `python -m activity.collect` first, or use the rule-based fallback.")
    for lab, ps in files.items():
        print(f"{lab:>9}: {', '.join(p.name for p in ps)}")

    names = feature_names(cfg)
    F, y, groups, _ = build_dataset(files, cfg, args.blocks)
    print(f"\n{len(F)} windows ({cfg.window_s}s, {cfg.overlap:.0%} overlap), {len(names)} features, "
          f"{len(np.unique(groups))} sessions")
    print("windows per class:", {a: int((y == a).sum()) for a in ACTIVITIES})

    standing_ok = np.mean([is_standing(f, cfg) for f, l in zip(F, y) if l == "standing"])
    moving_flagged = np.mean([is_standing(f, cfg) for f, l in zip(F, y) if l != "standing"])
    print(f"stage-1 rule: {standing_ok:.1%} of standing windows detected; "
          f"{moving_flagged:.1%} of moving windows wrongly flagged as standing")

    y_true, y_hat, y_tree = evaluate(F, y, groups, names, cfg)
    report("Leave-one-session-out: hybrid (rule + random forest)", y_true, y_hat)
    report("Baseline: rule-based tree (thresholds were set by eye from these same files, so "
           "this is optimistic)", y_true, y_tree)

    # Final model on everything
    train_mask = (y != "standing") & np.array([not is_standing(f, cfg) for f in F])
    model = make_forest().fit(F_mat(F, train_mask, names), y[train_mask])
    print("=== Random forest feature importances (final model) ===")
    for imp, n in sorted(zip(model.feature_importances_, names), reverse=True):
        print(f"  {n:<20} {imp:.3f}  {'#' * int(imp * 100)}")

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": names, "classes": list(model.classes_)},
                args.out / "model.joblib")
    cfg.save(args.out / "config.json")
    print(f"\nSaved {args.out / 'model.joblib'} and {args.out / 'config.json'}")


if __name__ == "__main__":
    main()
