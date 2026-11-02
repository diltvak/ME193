"""Real-time step and distance counter from the Pico's IMU stream.

    python -m step_nn.live --ip 10.247.137.46                 # Pico over Wi-Fi (console)
    python -m step_nn.live --ip 10.247.137.46 --plot          # + live plot window (for the video)
    python -m step_nn.live --source replay --file 33_steps    # replay a recording as if live
    python -m step_nn.live --source replay --file 33_steps --plot --speed 3

Close pc_receiver.py first (the Pico streams to one PC at a time). The Pico keeps running the
unchanged pico_main.py: it samples the MPU6050 at 100 Hz and broadcasts UDP packets; all the
machine learning runs here on the PC.

Plot window keys: r = reset counts (start a new walk), q = quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from activity.data_io import ReplaySource, UdpSource

from .config import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, StepConfig
from .counter import RidgeLength, StepCounter
from .model import load_net
from . import uncertainty


def build_counter(model_dir: Path):
    cfg = StepConfig.load_or_default(model_dir)
    if not (model_dir / "net.pt").exists():
        sys.exit(f"No trained model in {model_dir}. Run:  python -m step_nn.train")
    net = load_net(model_dir / "net.pt", cfg)
    ridge = RidgeLength.load(model_dir / "ridge.npz") if (model_dir / "ridge.npz").exists() else None
    return cfg, StepCounter(cfg, net, ridge)


def make_source(args, cfg):
    if args.source == "udp":
        return UdpSource(cfg, args.ip)
    files = [("walk", Path(args.data_dir) / f"{f}.csv") for f in args.file]
    for _, p in files:
        if not p.exists():
            sys.exit(f"replay file not found: {p}")
    return ReplaySource(cfg, files, speed=args.speed)


def report(counter, start, note=""):
    """Print one line per stride from index `start`, with the running totals as of that stride."""
    cum = 0.0
    for k, e in enumerate(counter.events, 1):
        cum += e["length"]
        if k > start:
            steps = 2 * k - (1 if counter.cfg.first_step_on_sensor_leg else 0)
            flag = "  (low confidence)" if e["conf"] < counter.cfg.weak_conf else ""
            print(f"stride {k:3d} | {steps:3d} steps | +{e['length']:.2f} m | total {cum:6.2f} m | p={e['conf']:.2f}{flag}{note}")
    return len(counter.events)


class LivePlot:
    """Top: last window of gz and the net's step probability. Big text: steps / distance."""

    def __init__(self, cfg, counter):
        import matplotlib.pyplot as plt
        self.plt, self.cfg, self.counter, self.quit = plt, cfg, counter, False
        self.unc = uncertainty.load()
        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1.5, 2.2])
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "quit", True))
        self.ax.axis("off")
        self.big = self.ax.text(0.30, 0.62, "", ha="center", va="center", fontsize=40, weight="bold",
                                transform=self.ax.transAxes)
        self.unc_txt = self.ax.text(0.5, 0.2, "", ha="center", va="center", fontsize=13, color="C3",
                                    transform=self.ax.transAxes)
        self.sub = self.ax.text(0.5, 0.0, "", ha="center", va="bottom", fontsize=11, transform=self.ax.transAxes)
        # distance gauge: right of the big number; bands = 95% / 68% interval, dot = estimate
        self.gauge = self.fig.add_axes([0.60, 0.74, 0.36, 0.11])
        self.gauge.set_facecolor("none")
        for sp in ("top", "right", "left"):
            self.gauge.spines[sp].set_visible(False)
        self.gauge.set_yticks([])
        self.l_gz, = self.ax2.plot([], [], lw=0.8, color="0.55", label="gz / 500 dps (ankle swing)")
        self.l_p, = self.ax2.plot([], [], lw=1.6, color="C0", label="neural net: step probability")
        self.l_ev, = self.ax2.plot([], [], "v", color="C3", ms=9, label="counted stride (confident)")
        self.l_weak, = self.ax2.plot([], [], "v", mfc="none", mec="C1", mew=2, ms=9,
                                     label="counted, low confidence")
        self.l_maybe, = self.ax2.plot([], [], "D", mfc="none", mec="0.4", ms=7, label="possible missed stride")
        self.ax2.axhline(cfg.peak_thresh, color="C0", ls=":", lw=0.8)
        self.ax2.set_ylim(-0.6, 1.3)
        self.ax2.set_xlabel("time (s)")
        self.ax2.legend(loc="lower left", fontsize=8)
        plt.ion()
        plt.show(block=False)

    def _key(self, e):
        if e.key == "r":
            self.counter.reset()
        elif e.key == "q":
            self.quit = True

    def _draw_gauge(self, c):
        g = self.gauge
        g.clear()
        g.set_yticks([])
        for sp in ("top", "right", "left"):
            g.spines[sp].set_visible(False)
        d = c.distance_m
        if c.strides == 0:
            g.set_xticks([])
            return
        a, b = uncertainty.distance_interval(d, self.unc)
        m, n = uncertainty.distance_interval(d, self.unc, "std")
        pad = 0.08 * (b - a)
        g.set_xlim(a - pad, b + pad)
        g.set_ylim(-1, 1.3)
        g.barh(0, b - a, left=a, height=0.7, color="C3", alpha=0.22, lw=0)
        g.barh(0, n - m, left=m, height=0.7, color="C3", alpha=0.45, lw=0)
        g.plot([d], [0], "o", color="k", ms=8, zorder=3)
        g.plot([a, a], [-0.35, 0.35], color="C3", lw=2)
        g.plot([b, b], [-0.35, 0.35], color="C3", lw=2)
        g.text((a + b) / 2, 0.55, f"\u00b1{(n - m) / 2:.1f} m (68%)   \u00b1{(b - a) / 2:.1f} m (95%)",
               ha="center", va="bottom", fontsize=11, color="C3", weight="bold")
        g.set_xticks([a, m, d, n, b])
        g.set_xticklabels([f"{v:.1f}" for v in (a, m, d, n, b)], fontsize=8)
        g.tick_params(axis="x", length=3)

    def _unc_text(self, c):
        if c.strides == 0:
            return ""
        u = self.unc
        lo, hi = c.steps_range
        step_rng = f"{lo}-{hi}" if (lo, hi) != (c.steps, c.steps) else "no doubtful strides"
        return (f"Steps: typical error \u00b1{u['step_max_err']} (worst held-out), from this signal: {step_rng}"
                f"  |  low-confidence strides: {c.weak_strides}")

    def draw(self, status):
        c = self.counter
        tu, prob, X = c.last_prob
        if len(tu):
            self.l_gz.set_data(tu, X[:, 5] / 500)
            self.l_p.set_data(tu, prob)
            vis = [e for e in c.events if e["t"] >= tu[0]]
            strong = [e["t"] for e in vis if e["conf"] >= self.cfg.weak_conf]
            weak = [e["t"] for e in vis if e["conf"] < self.cfg.weak_conf]
            maybe = [m["t"] for m in c.maybe if m["t"] >= tu[0]]
            self.l_ev.set_data(strong, np.full(len(strong), 1.15))
            self.l_weak.set_data(weak, np.full(len(weak), 1.15))
            self.l_maybe.set_data(maybe, np.full(len(maybe), 1.15))
            self.ax2.set_xlim(tu[0], tu[0] + self.cfg.infer_window_s)
        self.big.set_text(f"{c.steps} steps   {c.distance_m:.1f} m")
        self.unc_txt.set_text(self._unc_text(c))
        self._draw_gauge(c)
        cad = f"{c.cadence_spm:.0f} steps/min" if c.cadence_spm else "-"
        self.sub.set_text(f"{c.strides} strides of the sensor leg  |  cadence {cad}  |  {status}   "
                          f"[r = reset, q = quit]")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def print_final(counter):
    u = uncertainty.load()
    lo, hi = counter.steps_range
    d1, d2 = uncertainty.distance_interval(counter.distance_m, u, "std")
    a, b = uncertainty.distance_interval(counter.distance_m, u)
    print(f"\nFinal: {counter.steps} steps ({counter.strides} strides), {counter.distance_m:.1f} m")
    print(f"  steps:    typical error +/-{u['step_max_err']} (worst held-out); range from this signal {lo}-{hi} "
          f"({counter.weak_strides} low-confidence, {len(counter.maybe)} possible missed)")
    print(f"  distance: 68% {d1:.1f}-{d2:.1f} m, 95% {a:.1f}-{b:.1f} m (leave-one-trial-out, ~19.5 m walks)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["udp", "replay"], default="udp")
    ap.add_argument("--ip", default=StepConfig().pico_ip, help="Pico's IP address")
    ap.add_argument("--file", nargs="+", default=["33_steps"], help="replay: CSV stem(s) in --data-dir")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--plot", action="store_true", help="show the live plot window (needs matplotlib)")
    args = ap.parse_args()

    cfg, counter = build_counter(args.model_dir)
    source = make_source(args, cfg)
    plot = LivePlot(cfg, counter) if args.plot else None
    source.start()
    print(f"Source: {args.source} | stride length from: {cfg.len_source} | Ctrl-C to stop")

    seen, was_connected, last_draw, flushed = 0, None, 0.0, False
    try:
        while not (plot and plot.quit):
            t, X = source.read()
            if len(t):
                counter.update(t, X)
            if len(counter.events) < seen:                     # counts were reset from the plot window
                seen = 0
            seen = report(counter, seen)
            st = source.status()
            if st["connected"] != was_connected:
                was_connected = st["connected"]
                print("connected" if was_connected else "waiting for the Pico...")
            if not st["connected"] and args.source == "replay" and not flushed:
                counter.flush()
                flushed = True
                seen = report(counter, seen, "  (end of recording)")
                print(f"Done: {counter.steps} steps, {counter.distance_m:.1f} m")
                if not plot:
                    break
            if plot and time.time() - last_draw > 0.05:
                plot.draw("connected" if st["connected"] else "waiting / finished")
                last_draw = time.time()
            time.sleep(0.02 if plot else 0.05)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        print_final(counter)


if __name__ == "__main__":
    main()
