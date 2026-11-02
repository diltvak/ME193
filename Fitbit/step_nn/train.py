"""Train the final model on every labelled recording and save it to models/.

    python -m step_nn.train                       # uses results/results.json to pick the length source
    python -m step_nn.train --len-source ridge

Trains on the 10 distance trials, your three hand-counted walks, the extra walking file, and the
standing/cycling negatives. Saves net.pt (PyTorch weights), ridge.npz (baseline stride-length
model) and config.json (the exact StepConfig, so live.py matches). Run `python -m step_nn.evaluate`
first for honest accuracy numbers: this final model has seen every file, so scoring it on them
would be meaningless.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .config import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, DEFAULT_RESULTS_DIR, StepConfig
from .counter import RidgeLength, StepCounter, run_recording
from .data import load_all
from .model import save_net, train_net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--iters", type=int, default=StepConfig().iters)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--len-source", choices=["nn", "ridge"], default=None,
                    help="stride-length source; default: the best one in results/results.json, else nn")
    args = ap.parse_args()

    cfg = StepConfig(iters=args.iters)
    if args.len_source is None:
        res = DEFAULT_RESULTS_DIR / "results.json"
        if res.exists():
            args.len_source = json.loads(res.read_text())["loto_summary"]["best_len_source"]
    cfg = dataclasses.replace(cfg, len_source=args.len_source or "nn")

    data = load_all(args.data_dir, cfg)
    pos = data["trials"] + data["counts"] + data["extra"]
    print(f"Training on {len(pos)} walking + {len(data['negatives'])} non-walking recordings "
          f"({sum(len(r['peaks']) for r in pos)} labelled strides), {cfg.iters} iterations")
    net = train_net(pos, data["negatives"], cfg, seed=args.seed, verbose=True)
    ridge = RidgeLength.fit(data["trials"], cfg)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    save_net(net, args.model_dir / "net.pt")
    ridge.save(args.model_dir / "ridge.npz")
    cfg.save(args.model_dir / "config.json")
    print(f"Saved net.pt, ridge.npz, config.json to {args.model_dir} (stride length from: {cfg.len_source})")

    print("\nSanity check on the training files (optimistic - these were seen in training):")
    counter = StepCounter(cfg, net, ridge)
    for rec in data["counts"] + data["trials"][:3]:
        run_recording(counter, rec["raw_t"], rec["raw_X7"])
        truth = f"hand count {rec['true_steps']}" if "true_steps" in rec else "64 ft = 19.5 m"
        print(f"  {rec['name']:9s} counted {counter.steps:2d} steps, {counter.distance_m:5.1f} m   ({truth})")


if __name__ == "__main__":
    main()
