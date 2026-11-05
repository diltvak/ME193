"""Uncertainty analysis: how good is the counter, measured on data the model never trained on?

    python -m step_nn.evaluate                 # ~15 min on a laptop CPU
    python -m step_nn.evaluate --iters 150     # quick smoke test

Two experiments, both run through the same StepCounter that live.py uses (streaming, with latency,
dropped packets and all):

1. HELD-OUT STEP COUNT (ground truth = your hand counts). Train on the 10 distance trials (+ the
   walking/standing/cycling extras), test on 33_steps / 29_steps / 23_steps. Repeated over
   several random seeds to show run-to-run variation.
2. LEAVE-ONE-TRIAL-OUT (LOTO). For each of the 10 trials: train on the other 9 (+ the three
   hand-counted walks and extras), then count strides and estimate distance on the held-out trial.
   Ground truth is 64 ft and the stride labels (which the held-out walks showed to be exact).

Writes results/results.json and results/*.png. The stride-length source with the lower LOTO distance
error ("nn" or "ridge") is recorded as best_len_source and picked up by train.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, TRIAL_DISTANCE_M, StepConfig
from .counter import RidgeLength, StepCounter, run_recording
from .data import load_all
from .model import predict, train_net

MATCH_TOL_S = 0.15


def match_events(det_t, true_t, tol=MATCH_TOL_S):
    """Greedy one-to-one match of detected to labelled stride times -> (TP, FP, FN, timing errors)."""
    det_t, true_t = np.sort(det_t), np.sort(true_t)
    used, errs = set(), []
    for d in det_t:
        if len(true_t) == 0:
            break
        j = int(np.argmin(np.abs(true_t - d)))
        if abs(true_t[j] - d) <= tol and j not in used:
            used.add(j)
            errs.append(d - true_t[j])
    tp = len(used)
    return tp, len(det_t) - tp, len(true_t) - tp, errs


def lens(counter):
    return (sum(e["nn_length"] for e in counter.events),
            sum(e["ridge_length"] for e in counter.events) if counter.ridge else float("nan"))


def evaluate(args):
    cfg = StepConfig(iters=args.iters)
    data = load_all(args.data_dir, cfg)
    trials, counts, extra, negs = data["trials"], data["counts"], data["extra"], data["negatives"]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"config": {"iters": cfg.iters, "seeds": args.seeds}, "heldout": [], "loto": []}
    t_start = time.time()

    # ---------------- 1. held-out step counts -----------------------------------------------
    print(f"== Held-out step counts (train on trials{' + extras' if extra else ''}, test on your hand counts)")
    ridge = RidgeLength.fit(trials, cfg)
    plot_net = None
    for seed in range(args.seeds):
        net = train_net(trials + extra, negs, cfg, seed=seed)
        plot_net = net if plot_net is None else plot_net
        for rec in counts:
            c = run_recording(StepCounter(cfg, net, ridge), rec["raw_t"], rec["raw_X7"])
            nn_d, rd_d = lens(c)
            row = {"seed": seed, "name": rec["name"], "true_steps": rec["true_steps"], "strides": c.strides,
                   "steps": c.steps, "error": c.steps - rec["true_steps"], "dist_nn_m": nn_d, "dist_ridge_m": rd_d}
            res["heldout"].append(row)
            print(f"  seed {seed}  {rec['name']:9s} true {rec['true_steps']:2d}  counted {c.steps:2d}  "
                  f"(strides {c.strides}, error {row['error']:+d})   [{time.time() - t_start:.0f}s]")
    errs = np.array([r["error"] for r in res["heldout"]])
    res["heldout_summary"] = {"mean_abs_error_steps": float(np.abs(errs).mean()), "bias_steps": float(errs.mean()),
                              "exact_fraction": float((errs == 0).mean()),
                              "within_2_steps_fraction": float((np.abs(errs) <= 2).mean()),
                              "mean_abs_pct_error": float(np.mean([abs(r["error"]) / r["true_steps"] * 100
                                                                   for r in res["heldout"]]))}

    # ---------------- 2. leave-one-trial-out ------------------------------------------------
    print("== Leave-one-trial-out (stride count vs labels, distance vs 64 ft)")
    cv_plot = {}
    for i, rec in enumerate(trials):
        rest = trials[:i] + trials[i + 1:]
        net = train_net(rest + counts + extra, negs, cfg, seed=100 + i)
        ridge = RidgeLength.fit(rest, cfg)
        c = run_recording(StepCounter(cfg, net, ridge), rec["raw_t"], rec["raw_X7"])
        nn_d, rd_d = lens(c)
        tp, fp, fn, terr = match_events([e["t"] for e in c.events], rec["t"][rec["peaks"]])
        row = {"name": rec["name"], "labelled_strides": int(len(rec["peaks"])), "strides": c.strides, "tp": tp,
               "fp": fp, "fn": fn, "timing_err_s": float(np.mean(np.abs(terr))) if terr else None,
               "dist_nn_m": nn_d, "dist_ridge_m": rd_d,
               "err_nn_pct": (nn_d - TRIAL_DISTANCE_M) / TRIAL_DISTANCE_M * 100,
               "err_ridge_pct": (rd_d - TRIAL_DISTANCE_M) / TRIAL_DISTANCE_M * 100}
        res["loto"].append(row)
        print(f"  {rec['name']:9s} strides {c.strides:2d}/{row['labelled_strides']:2d} (TP {tp} FP {fp} FN {fn})  "
              f"dist nn {nn_d:5.1f} m ({row['err_nn_pct']:+5.1f}%)  ridge {rd_d:5.1f} m ({row['err_ridge_pct']:+5.1f}%)"
              f"   [{time.time() - t_start:.0f}s]")
        cv_plot[rec["name"]] = row

    # ---------------- summary statistics ----------------------------------------------------
    L = res["loto"]
    sc = np.array([r["strides"] - r["labelled_strides"] for r in L])
    tp, fp, fn = (sum(r[k] for r in L) for k in ("tp", "fp", "fn"))
    summary = {"stride_count_mean_abs_error": float(np.abs(sc).mean()),
               "stride_count_bias": float(sc.mean()),
               "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
               "f1": 2 * tp / max(1, 2 * tp + fp + fn),
               "mean_timing_error_s": float(np.mean([r["timing_err_s"] for r in L if r["timing_err_s"] is not None]))}
    rng = np.random.default_rng(0)
    for src in ("nn", "ridge"):
        e = np.array([r[f"err_{src}_pct"] for r in L])
        boot = [np.abs(rng.choice(e, len(e))).mean() for _ in range(5000)]
        tcrit = stats.t.ppf(0.975, len(e) - 1)
        summary[src] = {"mean_abs_pct": float(np.abs(e).mean()), "bias_pct": float(e.mean()),
                        "std_pct": float(e.std(ddof=1)), "rmse_pct": float(np.sqrt((e ** 2).mean())),
                        "worst_abs_pct": float(np.abs(e).max()),
                        "mean_abs_pct_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "mean_signed_pct_ci95": [float(e.mean() - tcrit * e.std(ddof=1) / np.sqrt(len(e))),
                                                 float(e.mean() + tcrit * e.std(ddof=1) / np.sqrt(len(e)))],
                        "pred_interval95_pct": [float(e.mean() - tcrit * e.std(ddof=1) * np.sqrt(1 + 1 / len(e))),
                                                float(e.mean() + tcrit * e.std(ddof=1) * np.sqrt(1 + 1 / len(e)))]}
    summary["best_len_source"] = min(("nn", "ridge"), key=lambda s: summary[s]["rmse_pct"])
    res["loto_summary"] = summary
    res["runtime_s"] = time.time() - t_start

    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print("\n== Summary")
    print("  held-out step count:", json.dumps(res["heldout_summary"]))
    print("  LOTO strides:", {k: round(v, 3) for k, v in summary.items() if not isinstance(v, (dict, str))})
    for src in ("nn", "ridge"):
        print(f"  LOTO distance [{src}]:", json.dumps({k: (np.round(v, 2).tolist()) for k, v in summary[src].items()}))
    print("  best stride-length source:", summary["best_len_source"])

    make_plots(res, counts, trials, plot_net, cfg, out_dir)
    print(f"Wrote {out_dir / 'results.json'} and plots ({res['runtime_s']:.0f}s)")


def make_plots(res, counts, trials, net, cfg, out_dir):
    # --- held-out recordings: probability trace, detections, labels ---------------------------
    fig, axs = plt.subplots(len(counts), 1, figsize=(11, 2.9 * len(counts)))
    ridge = None
    for ax, rec in zip(np.atleast_1d(axs), counts):
        prob, _ = predict(net, rec["X"])
        c = run_recording(StepCounter(cfg, net, ridge), rec["raw_t"], rec["raw_X7"])
        gz = rec["X"][:, 5]
        ax.plot(rec["t"], gz / 500, lw=0.7, color="0.6", label="gz / 500 dps")
        ax.plot(rec["t"], prob, lw=1.2, color="C0", label="net: step probability")
        ax.plot(rec["t"][rec["peaks"]], np.full(len(rec["peaks"]), 1.08), "o", color="C2", ms=5,
                label=f"labelled strides ({len(rec['peaks'])})")
        ax.plot([e["t"] for e in c.events], np.full(c.strides, 1.18), "v", color="C3", ms=6,
                label=f"counted strides ({c.strides})")
        ax.axhline(cfg.peak_thresh, color="C0", ls=":", lw=0.8)
        ax.set_title(f"{rec['name']}: you counted {rec['true_steps']} steps; net counted {c.steps} "
                     f"(2 x {c.strides} - 1)", fontsize=10)
        ax.set_ylim(-0.6, 1.3)
        ax.set_xlabel("time (s)")
        ax.legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "heldout_steps.png", dpi=110)
    plt.close()

    # --- LOTO: stride counts and distance -----------------------------------------------------
    L = res["loto"]
    names = [r["name"].replace("trial_", "") for r in L]
    x = np.arange(len(L))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot([8, 21], [8, 21], "k--", lw=0.8)
    axs[0].scatter([r["labelled_strides"] for r in L], [r["strides"] for r in L], c="C0")
    axs[0].set_xlabel("labelled strides")
    axs[0].set_ylabel("counted strides (held-out net)")
    axs[0].set_title("Stride count, leave-one-trial-out")
    axs[1].bar(x - 0.2, [r["dist_nn_m"] for r in L], 0.4, label="neural net head", color="C0")
    axs[1].bar(x + 0.2, [r["dist_ridge_m"] for r in L], 0.4, label="ridge baseline", color="C1")
    axs[1].axhline(TRIAL_DISTANCE_M, color="k", ls="--", lw=1, label="truth: 64 ft = 19.5 m")
    axs[1].set_xticks(x, names)
    axs[1].set_xlabel("trial")
    axs[1].set_ylabel("estimated distance (m)")
    axs[1].set_title("Distance, leave-one-trial-out")
    axs[1].legend(fontsize=8)
    axs[2].bar(x - 0.2, [r["err_nn_pct"] for r in L], 0.4, color="C0")
    axs[2].bar(x + 0.2, [r["err_ridge_pct"] for r in L], 0.4, color="C1")
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].set_xticks(x, names)
    axs[2].set_xlabel("trial")
    axs[2].set_ylabel("distance error (%)")
    axs[2].set_title("Distance error")
    plt.tight_layout()
    plt.savefig(out_dir / "loto_summary.png", dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--iters", type=int, default=StepConfig().iters)
    ap.add_argument("--seeds", type=int, default=3, help="seeds for the held-out step-count test")
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()
