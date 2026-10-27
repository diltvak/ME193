"""Real-time activity recognition from the live stream.

    python -m activity.live --ip 10.247.137.46            # Pico over Wi-Fi
    python -m activity.live --source sim --plot           # simulated data + live plot
    python -m activity.live --source replay --plot        # replay your recorded CSVs as "live"
    python -m activity.live --rules                       # ignore the trained model, use the rule tree
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .classifier import HybridClassifier, MajorityVote
from .config import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, Config
from .data_io import discover_files, make_source, resample_uniform


class Engine:
    """Rolling buffer -> fixed-rate window -> features -> classifier -> majority vote.

    Call step() often (every ~50 ms). It returns None until a new window is due, then an
    event dict: status ('ok' | 'warming' | 'gap' | 'disconnected'), raw/smooth labels, features.
    """

    def __init__(self, cfg: Config, source, clf: HybridClassifier):
        self.cfg, self.source, self.clf = cfg, source, clf
        self.vote = MajorityVote(cfg.vote_windows)
        self.t = np.empty(0)
        self.X = np.empty((0, 7))
        self.next_eval = None
        self._was_connected = True
        self._ever_connected = False

    def _append(self, t, X):
        self.t = np.concatenate([self.t, t])
        self.X = np.vstack([self.X, X])
        keep = self.t > self.t[-1] - self.cfg.buffer_s
        self.t, self.X = self.t[keep], self.X[keep]

    def step(self):
        cfg = self.cfg
        t, X = self.source.read()
        if len(t):
            self._append(t, X)
        st = self.source.status()
        if not st["connected"]:
            if self._was_connected:                       # edge: just lost the stream
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
        if not valid[-n:].all():                          # long dropout inside the window
            self.vote.reset()
            return {"status": "gap", "stats": st}
        win = Xu[-n:]
        raw, conf, feats = self.clf.predict_window(win)

        # Fast path into standing: a full window is still half-moving for ~2 s after you stop,
        # and that fading motion looks smooth and low-energy - i.e. like biking. Checking only
        # the most recent second catches the stop right away and flushes the vote history.
        # The same partial-window effect happens when you start moving, so a window that *begins*
        # still is a transition: it is shown but does not vote.
        q = int(cfg.quick_stand_s * cfg.fs)
        if self._still(win[-q:]):
            raw, conf = "standing", 1.0
            self.vote.reset("standing")
            smooth = "standing"
        elif self._still(win[:q]) or conf < cfg.min_confidence:
            smooth = self.vote.current() or raw           # transition / uncertain: don't vote
            raw = f"{raw}?"
        else:
            smooth = self.vote.update(raw)
        return {"status": "ok", "t": t_end, "raw": raw, "conf": conf, "smooth": smooth,
                "feats": feats, "stats": st}

    def _still(self, seg):
        cfg = self.cfg
        return (np.linalg.norm(seg[:, :3], axis=1).std() < cfg.stand_acc_std_g
                and np.linalg.norm(seg[:, 3:6], axis=1).std() < cfg.stand_gyro_std_dps)

    def recent(self, seconds):
        m = self.t >= (self.t[-1] - seconds) if len(self.t) else slice(0)
        return self.t[m], self.X[m]


def format_event(ev, truth=None):
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
    f = ev["feats"]
    s = (f"[{ev['t']:7.1f}s] {ev['smooth'].upper():<9} (raw {ev['raw']:<8}) "
         f"|acc| std {f['a_std']:.2f} g  f0 {f['g1_f0']:.2f} Hz  sinus {f['g1_fund_frac']:.2f}  {link}")
    return s + (f"  truth={truth}" if truth else "")


def run_console(engine):
    last_status = None
    while True:
        ev = engine.step()
        if ev is not None and not (ev["status"] != "ok" and ev["status"] == last_status):
            print(format_event(ev, engine.source.truth), flush=True)
            last_status = ev["status"]
        time.sleep(0.05)


# Everything the IMU streams, plus the two magnitudes the classifier works on.
# name -> function(X) where X is the (n, 7) buffer: ax ay az gx gy gz temp
CHANNELS = {
    "|acc| (g)": lambda X: np.linalg.norm(X[:, :3], axis=1),
    "|gyro| (dps)": lambda X: np.linalg.norm(X[:, 3:6], axis=1),
    "ax (g)": lambda X: X[:, 0],
    "ay (g)": lambda X: X[:, 1],
    "az (g)": lambda X: X[:, 2],
    "gx (dps)": lambda X: X[:, 3],
    "gy (dps)": lambda X: X[:, 4],
    "gz (dps)": lambda X: X[:, 5],
    "temp (C)": lambda X: X[:, 6],
}


def run_plot(engine, seconds=10, max_seconds=None):
    """Qt window with two live plots; each has a dropdown to pick which IMU channel it shows.

    max_seconds is only for automated tests (closes the window after that long).
    """
    from PyQt6 import QtCore, QtWidgets                       # import first so matplotlib picks PyQt6
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    colors = {"standing": "#7f7f7f", "walking": "#2ca02c", "running": "#d62728", "biking": "#1f77b4"}
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setWindowTitle("Activity monitor")
    layout = QtWidgets.QVBoxLayout(win)

    label = QtWidgets.QLabel("waiting for data...")
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("font-size: 28px; font-weight: bold;")
    layout.addWidget(label)

    fig = Figure(figsize=(9, 6), tight_layout=True)
    canvas = FigureCanvasQTAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    axes[1].set_xlabel("time (s)")
    panels = []                                                # (combo, axis, line)
    for ax, default in zip(axes, ("|acc| (g)", "|gyro| (dps)")):
        combo = QtWidgets.QComboBox()
        combo.addItems(list(CHANNELS))
        combo.setCurrentText(default)
        (line,) = ax.plot([], [], lw=1, color="tab:blue" if default.startswith("|acc") else "tab:orange")
        ax.set_ylabel(default)
        ax.grid(True, alpha=0.3)
        combo.currentTextChanged.connect(lambda name, ax=ax: ax.set_ylabel(name))
        panels.append((combo, ax, line))
    row = QtWidgets.QHBoxLayout()
    for i, (combo, _, _) in enumerate(panels):
        row.addWidget(QtWidgets.QLabel(f"Top plot:" if i == 0 else "Bottom plot:"))
        row.addWidget(combo, 1)
    layout.addLayout(row)
    layout.addWidget(canvas, 1)

    def tick():
        ev = engine.step()
        if ev is not None:
            print(format_event(ev, engine.source.truth), flush=True)
            if ev["status"] == "ok":
                label.setText(f"{ev['smooth'].upper()}   (raw: {ev['raw']})")
                label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {colors.get(ev['smooth'], 'black')};")
            elif ev["status"] == "disconnected":
                label.setText("DISCONNECTED - waiting for stream")
                label.setStyleSheet("font-size: 28px; font-weight: bold; color: black;")
        t, X = engine.recent(seconds)
        if len(t) > 2:
            for combo, ax, line in panels:
                line.set_data(t, CHANNELS[combo.currentText()](X))
                ax.relim()
                ax.autoscale_view(scalex=False)
            axes[0].set_xlim(t[-1] - seconds, t[-1])
            canvas.draw_idle()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(50)
    if max_seconds:
        QtCore.QTimer.singleShot(int(max_seconds * 1000), app.quit)
    win.resize(1000, 700)
    win.show()
    app.exec()
    win.canvas = canvas                                        # keep refs alive until exit
    return win


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["udp", "sim", "replay"], default="udp")
    ap.add_argument("--ip", help="Pico IP (default from config.py)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--rules", action="store_true", help="use the rule-based tree instead of the model")
    ap.add_argument("--plot", action="store_true", help="show a live matplotlib plot")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed for sim/replay")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="CSVs for --source replay")
    args = ap.parse_args(argv)

    clf = HybridClassifier.load(args.model_dir, force_rules=args.rules)
    cfg = clf.cfg
    if args.ip:
        cfg.pico_ip = args.ip
    replay = None
    if args.source == "replay":
        found = discover_files(args.data_dir)
        replay = [(lab, ps[0]) for lab, ps in sorted(found.items())]
        if not replay:
            sys.exit(f"no *_data_*.csv files in {args.data_dir}")
    source = make_source(args.source, cfg, ip=args.ip, replay_files=replay, speed=args.speed)
    engine = Engine(cfg, source, clf)

    print(f"source={args.source}  classifier={'random forest' if clf.uses_model else 'rule-based tree'}  "
          f"window={cfg.window_s}s hop={cfg.live_hop_s:.2f}s vote={cfg.vote_windows}   (Ctrl+C to quit)")
    source.start()
    try:
        (run_plot if args.plot else run_console)(engine)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        source.stop()


if __name__ == "__main__":
    main()
