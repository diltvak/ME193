"""Guided data collection: prompts for each activity, records the live stream, saves labeled CSVs.

    python -m activity.collect --ip 10.247.137.46 --duration 60
    python -m activity.collect --source sim --duration 10          # no hardware

Files are written as <label>_data_<n>.csv in the same format as pc_receiver.py, and are
picked up automatically by `python -m activity.train`. Close pc_receiver.py first (the Pico
streams to one PC at a time).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import ACTIVITIES, DEFAULT_DATA_DIR, ROOT, Config
from .data_io import make_source, save_csv

HINTS = {
    "standing": "stand still",
    "walking": "walk at a normal pace",
    "running": "run (treadmill or outdoors)",
    "biking": "cycle, or lie on your back and air-bike",
}


def next_path(out_dir: Path, label: str) -> Path:
    n = 1
    while (out_dir / f"{label}_data_{n}.csv").exists():
        n += 1
    return out_dir / f"{label}_data_{n}.csv"


def record(source, seconds: float):
    """Read from `source` for `seconds` of wall-clock time; returns (t, X, pc_time) arrays."""
    ts, xs, pcs = [], [], []
    t_end = time.time() + seconds
    last_print = 0
    while time.time() < t_end:
        t, X = source.read()
        if len(t):
            ts.append(t), xs.append(X), pcs.append(np.full(len(t), time.time()))
        if time.time() - last_print >= 5:
            st = source.status()
            print(f"   {t_end - time.time():5.0f}s left | samples {st['samples']} | "
                  f"lost packets {st['lost']} | {'connected' if st['connected'] else 'NO DATA'}")
            last_print = time.time()
        time.sleep(0.02)
    if not ts:
        return np.empty(0), np.empty((0, 7)), np.empty(0)
    t = np.concatenate(ts)
    return t - t[0], np.vstack(xs), np.concatenate(pcs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["udp", "sim"], default="udp")
    ap.add_argument("--ip", help="Pico IP (default from config.py)")
    ap.add_argument("--duration", type=float, default=60, help="seconds per activity")
    ap.add_argument("--countdown", type=float, default=5, help="get-ready seconds before recording")
    ap.add_argument("--activities", nargs="+", default=ACTIVITIES, choices=ACTIVITIES)
    ap.add_argument("--out-dir", type=Path, help="default: project folder (sim: activity/sim_data)")
    args = ap.parse_args(argv)

    cfg = Config()
    out_dir = args.out_dir or (ROOT / "sim_data" if args.source == "sim" else DEFAULT_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label in args.activities:
        source = make_source(args.source, cfg, ip=args.ip,
                             schedule=[(label, args.duration + args.countdown + 60)])
        source.start()
        try:
            if args.source == "udp":
                print("Waiting for the Pico stream...", end="", flush=True)
                t0 = time.time()
                while not source.status()["connected"]:
                    if time.time() - t0 > 20:
                        raise SystemExit("\nNo data from the Pico. Check its IP, that it is on Wi-Fi, "
                                         "and that pc_receiver.py is closed.")
                    time.sleep(0.2)
                print(" connected.")
            input(f"\n[{label.upper()}] Get ready to {HINTS[label]} for {args.duration:.0f}s. "
                  "Press Enter to start the countdown...")
            for s in range(int(args.countdown), 0, -1):
                print(f"  starting in {s}...", flush=True)
                time.sleep(1)
                source.read()                                   # discard samples during countdown
            print(f"  >>> {label.upper()} NOW <<<")
            t, X, pc = record(source, args.duration)
        finally:
            source.stop()
        if len(t) < 10:
            print("  no data recorded; skipping")
            continue
        path = next_path(out_dir, label)
        save_csv(path, t, X, pc)
        print(f"  saved {len(t)} samples ({t[-1]:.1f}s, ~{len(t) / max(t[-1], 1e-6):.0f} Hz) -> {path}")
    print("\nDone. Train with: python -m activity.train")


if __name__ == "__main__":
    main()
