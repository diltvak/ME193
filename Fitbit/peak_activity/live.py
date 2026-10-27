"""Real-time activity recognition from one accelerometer axis.

    python -m peak_activity.live --ip 10.247.137.46     # Pico over Wi-Fi
    python -m peak_activity.live --source replay        # replay your recorded CSVs as if live
    python -m peak_activity.live --plot                 # + live plot window (needs PyQt6, matplotlib)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from activity.data_io import UdpSource, ReplaySource, discover_files, resample_uniform

from .config import ACTIVITIES, DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, PeakConfig
from .peak import MajorityVote, classify_peak, peak_accel, window_peak


class Engine:
    """Rolling buffer -> fixed-rate window -> peak on one axis -> thresholds -> majority vote.

    step() returns None until a new window is due, then an event dict with
    status ('ok' | 'warming' | 'gap' | 'disconnected'), raw/smooth labels and the peak.
    """

    def __init__(self, cfg: PeakConfig, source):
        self.cfg, self.source = cfg, source
        self.vote = MajorityVote(cfg.vote_windows)
        self.t = np.empty(0)
        self.X = np.empty((0, 7))
        self.next_eval = None
        self._was_connected = True
        self._ever_connected = False

    def step(self):
        cfg = self.cfg
        t, X = self.source.read()
        if len(t):
            self.t, self.X = np.concatenate([self.t, t]), np.vstack([self.X, X])
            keep = self.t > self.t[-1] - cfg.buffer_s
            self.t, self.X = self.t[keep], self.X[keep]
        st = self.source.status()
        if not st["connected"]:
            if self._was_connected:
                self._was_connected = False
                self.vote.reset()
                self.next_eval = None
                return {"status": "disconnected", "stats": st, "ever": self._ever_connected}
            return None
        self._was_connected = self._ever_connected = True
        if len(self.t) < 2:
            return None
        t_end = self.t[-1]
        if self.next_eval is None:
            self.next_eval = self.t[0] + cfg.window_s
        if t_end < self.next_eval:
            return None
        self.next_eval = t_end + cfg.live_hop_s

        recent = self.t >= t_end - cfg.window_s - 0.1
        _, Xu, valid = resample_uniform(self.t[recent], self.X[recent], cfg)
        n = cfg.window_n
        if len(Xu) < n:
            return {"status": "warming", "stats": st}
        if not valid[-n:].all():
            self.vote.reset()
            return {"status": "gap", "stats": st}
        win = Xu[-n:]
        peak = window_peak(win, cfg)
        raw = classify_peak(peak, cfg)

        # Fast path into standing: a full window is still half-moving for ~2 s after you stop.
        # If the most recent second is flat, report standing now and flush the vote history.
        q = int(cfg.quick_stand_s * cfg.fs)
        if peak_accel(win[-q:, cfg.axis], cfg.peak_pct) < cfg.t_stand_bike:
            raw = "standing"
            self.vote.reset("standing")
            smooth = "standing"
        else:
            smooth = self.vote.update(raw)
        return {"status": "ok", "t": t_end, "raw": raw, "smooth": smooth, "peak": peak, "stats": st}


def format_event(ev, cfg, truth=None):
    st = ev["stats"]
    link = f"rx {st['samples']} lost {st['lost']} bad {st['bad']}"
    if ev["status"] == "disconnected":
        if not ev["ever"]:
            return "waiting for the Pico stream (is it on Wi-Fi? is pc_receiver.py closed?)..."
        return f"!! stream lost - waiting for the Pico to resume ({link})"
    if ev["status"] == "warming":
        return "filling first window..."
    if ev["status"] == "gap":
        return f"data gap in window, skipping ({link})"
    s = (f"[{ev['t']:7.1f}s] {ev['smooth'].upper():<9} (raw {ev['raw']:<8}) "
         f"peak a{cfg.axis_name} {ev['peak']:.2f} g  {link}")
    return s + (f"  truth={truth}" if truth else "")


COLORS = {"standing": "#7f7f7f", "biking": "#1f77b4", "walking": "#2ca02c", "running": "#d62728"}


def run_plot(engine, source, show_truth, seconds=10, history_s=60, max_seconds=None):
    """Qt window: big activity label, the accelerometer axis trace, and the peak history.

    The bottom plot is exactly what the classifier sees: one peak value per window, with the
    three thresholds drawn and the bands shaded in each activity's colour. The top plot has a
    dropdown to look at any accelerometer axis. max_seconds is only for automated tests.
    """
    from PyQt6 import QtCore, QtWidgets                       # import first so matplotlib picks PyQt6
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    cfg = engine.cfg
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setWindowTitle("Peak-acceleration activity monitor")
    layout = QtWidgets.QVBoxLayout(win)

    label = QtWidgets.QLabel("waiting for data...")
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("font-size: 28px; font-weight: bold;")
    layout.addWidget(label)

    fig = Figure(figsize=(9, 6), tight_layout=True)
    canvas = FigureCanvasQTAgg(fig)
    ax_sig, ax_pk = fig.subplots(2, 1)
    combo = QtWidgets.QComboBox()
    combo.addItems([f"a{n} (g)" for n in "xyz"])
    combo.setCurrentIndex(cfg.axis)
    row = QtWidgets.QHBoxLayout()
    row.addWidget(QtWidgets.QLabel("Top plot:"))
    row.addWidget(combo, 1)
    layout.addLayout(row)
    layout.addWidget(canvas, 1)

    (sig_line,) = ax_sig.plot([], [], lw=1, color="tab:blue")
    ax_sig.set_xlabel("time (s)")
    ax_sig.grid(True, alpha=0.3)
    combo.currentTextChanged.connect(lambda name: ax_sig.set_ylabel(name))
    ax_sig.set_ylabel(combo.currentText())

    top = max(cfg.t_walk_run * 2.0, 4.5)
    edges = [0.0, cfg.t_stand_bike, cfg.t_bike_walk, cfg.t_walk_run, top]
    for lab, lo, hi in zip(ACTIVITIES, edges, edges[1:]):
        ax_pk.axhspan(lo, hi, color=COLORS[lab], alpha=0.12)
        ax_pk.text(0.995, (lo + hi) / 2, lab, transform=ax_pk.get_yaxis_transform(), ha="right",
                   va="center", fontsize=9, color=COLORS[lab])
    for y in edges[1:-1]:
        ax_pk.axhline(y, color="k", lw=0.6, ls="--")
    (pk_line,) = ax_pk.plot([], [], "-", lw=1, color="0.4")
    pk_dots = ax_pk.scatter([], [], s=28, zorder=3)
    ax_pk.set_ylim(0, top)
    ax_pk.set_ylabel(f"peak of a{cfg.axis_name} (g)")
    ax_pk.set_xlabel("time (s)")
    ax_pk.grid(True, alpha=0.3)

    hist_t, hist_p, hist_c = [], [], []

    def tick():
        ev = engine.step()
        if ev is not None:
            print(format_event(ev, cfg, source.truth if show_truth else None), flush=True)
            if ev["status"] == "ok":
                label.setText(f"{ev['smooth'].upper()}   (raw: {ev['raw']})")
                label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS[ev['smooth']]};")
                hist_t.append(ev["t"]), hist_p.append(ev["peak"]), hist_c.append(COLORS[ev["raw"]])
            elif ev["status"] == "disconnected":
                label.setText("DISCONNECTED - waiting for stream")
                label.setStyleSheet("font-size: 28px; font-weight: bold; color: black;")
        if len(engine.t) > 2:
            t_end = engine.t[-1]
            m = engine.t >= t_end - seconds
            sig_line.set_data(engine.t[m], engine.X[m, combo.currentIndex()])
            ax_sig.relim()
            ax_sig.autoscale_view(scalex=False)
            ax_sig.set_xlim(t_end - seconds, t_end)
            while hist_t and hist_t[0] < t_end - history_s:
                hist_t.pop(0), hist_p.pop(0), hist_c.pop(0)
            pk_line.set_data(hist_t, hist_p)
            pk_dots.set_offsets(np.column_stack([hist_t, hist_p]) if hist_t else np.empty((0, 2)))
            pk_dots.set_color(hist_c)
            ax_pk.set_xlim(t_end - history_s, t_end)
            canvas.draw_idle()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(50)
    if max_seconds:
        QtCore.QTimer.singleShot(int(max_seconds * 1000), app.quit)
    win.resize(1000, 750)
    win.show()
    app.exec()
    win.canvas = canvas                                        # keep refs alive until exit
    return win


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["udp", "replay"], default="udp")
    ap.add_argument("--ip", help="Pico IP (default from config.py)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="CSVs for --source replay")
    ap.add_argument("--plot", action="store_true", help="show the live plot window")
    ap.add_argument("--plot-seconds", type=float, default=None, help=argparse.SUPPRESS)  # auto-close, for tests
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed for replay")
    args = ap.parse_args(argv)

    cfg = PeakConfig.load_or_default(args.model_dir)
    if args.ip:
        cfg.pico_ip = args.ip
    if args.source == "replay":
        found = discover_files(args.data_dir)
        files = [(lab, found[lab][0]) for lab in ACTIVITIES if lab in found]
        if not files:
            sys.exit(f"no *_data_*.csv files in {args.data_dir}")
        source = ReplaySource(cfg, files, speed=args.speed)
    else:
        source = UdpSource(cfg, args.ip)
    engine = Engine(cfg, source)

    print(f"source={args.source}  axis=a{cfg.axis_name}  standing<{cfg.t_stand_bike} biking<{cfg.t_bike_walk} "
          f"walking<{cfg.t_walk_run} running>= (g)  window={cfg.window_s}s vote={cfg.vote_windows}   (Ctrl+C to quit)")
    source.start()
    last_status = None
    try:
        if args.plot:
            run_plot(engine, source, args.source == "replay", max_seconds=args.plot_seconds)
            return
        while True:
            ev = engine.step()
            if ev is not None and not (ev["status"] != "ok" and ev["status"] == last_status):
                print(format_event(ev, cfg, source.truth if args.source == "replay" else None), flush=True)
                last_status = ev["status"]
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        source.stop()


if __name__ == "__main__":
    main()
