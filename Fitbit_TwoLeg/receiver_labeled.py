"""Two-leg receiver with route labels: mark Start, Waypoints 1-4 and End while you walk.

    Qt plugin workaround:
    chflags -R nohidden .venv && python3 receiver_labeled.py

    python receiver_labeled.py --left 10.0.0.251 --right 10.0.0.252
    python receiver_labeled.py --left 127.0.0.1:5006 --right 127.0.0.1:5007     # with fake_pico.py
    python receiver_labeled.py --rebuild recordings/route_20261120_110000      # redo labeled.csv

Walking a route:
  1. Connect both Picos, type a trial name (e.g. "route").
  2. Press Start (or Space) at the start point. This also starts recording.
  3. Press Space (or the Waypoint button) as you reach each waypoint, then at the end point.
     Labels can only be placed in order; "Undo last label" removes a mistake.
  4. End stops the recording 1.5 s later (untick "Auto-stop at End" to keep recording).

Each recording folder recordings/<trial>_<timestamp>/ contains:
  labeled.csv   both legs on one 100 Hz time grid + labels  <- the file for training
  left.csv, right.csv   raw samples per leg (same format as pc_receiver.py)
  labels.csv    t_s,label,pc_time: one row per button press

labeled.csv columns:
  t_s                       seconds since recording started (shared by both legs)
  left_ax..left_gz          left leg accel (g) and gyro (dps), interpolated onto the grid
  left_valid                1 if real left data is within 0.1 s, else 0 (values are blank then)
  right_ax..right_gz, right_valid
  label                     "start", "wp1".."wp4", "end" on the sample nearest the button press, else blank
  last_point, next_point    the route point most recently passed / coming up next
  segment                   0 before start, 1 = start->wp1, ..., 5 = wp4->end, 6 after end
  segment_progress          0..1 fraction of the time between last_point and next_point
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
import time
from pathlib import Path

import numpy as np

import receiver as base
from receiver import CHANNELS, LEGS, RECORD_DIR, Receiver, parse_address

POINTS = ["start", "wp1", "wp2", "wp3", "wp4", "end"]
POINT_NAMES = {"start": "Start", "wp1": "Waypoint 1", "wp2": "Waypoint 2", "wp3": "Waypoint 3",
               "wp4": "Waypoint 4", "end": "End"}
GRID_HZ = 100.0
MAX_GAP_S = 0.1             # longer holes in a leg's data are left blank (valid = 0) in labeled.csv
AUTO_STOP_DELAY_MS = 1500
IMU_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


# ----------------------------------------------------------------------------
# Building labeled.csv (also usable on its own: --rebuild)
# ----------------------------------------------------------------------------
def _load(path):
    try:
        d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError):
        return np.empty((0, 9))
    return d[np.isfinite(d).all(axis=1)] if len(d) else np.empty((0, 9))


def load_labels(folder):
    out = []
    p = Path(folder) / "labels.csv"
    if p.exists():
        for line in p.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 2 and parts[1] in POINTS:
                out.append((float(parts[0]), parts[1]))
    return sorted(out)


def _resample(raw, grid):
    """Raw leg rows (t_s, pc_time, ax..gz, temp) -> (len(grid), 6) values, valid mask."""
    Y = np.full((len(grid), 6), np.nan)
    valid = np.zeros(len(grid), bool)
    if len(raw) < 2:
        return Y, valid
    t = raw[:, 0]
    keep = np.concatenate([[True], np.diff(np.maximum.accumulate(t)) > 0])   # strictly increasing
    t, X = t[keep], raw[keep, 2:8]
    hi = np.searchsorted(t, grid, side="right")
    inside = (hi > 0) & (hi < len(t))
    hi = np.clip(hi, 1, len(t) - 1)
    valid = inside & ((t[hi] - t[hi - 1]) <= MAX_GAP_S)
    for c in range(6):
        Y[:, c] = np.interp(grid, t, X[:, c])
    Y[~valid] = np.nan
    return Y, valid


def build_labeled_csv(folder):
    """Merge left.csv, right.csv and labels.csv into labeled.csv. Returns a one-line summary."""
    folder = Path(folder)
    legs = {leg: _load(folder / f"{leg}.csv") for leg in LEGS}
    labels = load_labels(folder)
    ends = [d[-1, 0] for d in legs.values() if len(d)] + [t for t, _ in labels]
    if not ends:
        return f"{folder.name}: no data, labeled.csv not written"
    grid = np.arange(0.0, max(ends) + 0.5 / GRID_HZ, 1.0 / GRID_HZ)
    res = {leg: _resample(legs[leg], grid) for leg in LEGS}

    lab_t = np.array([t for t, _ in labels])
    lab_n = [n for _, n in labels]
    label_col = [""] * len(grid)
    for t, n in labels:
        i = int(np.clip(round(t * GRID_HZ), 0, len(grid) - 1))
        label_col[i] = n
    k = np.searchsorted(lab_t, grid, side="right")        # labels passed so far at each row

    header = (["t_s"] + [f"left_{c}" for c in IMU_COLS] + ["left_valid"]
              + [f"right_{c}" for c in IMU_COLS] + ["right_valid"]
              + ["label", "last_point", "next_point", "segment", "segment_progress"])

    def fmt(Y, v, i):
        if not v[i]:
            return ",,,,,,0"
        return ",".join(f"{x:.5f}" for x in Y[i]) + ",1"

    lines = [",".join(header)]
    for i, t in enumerate(grid):
        ki = int(k[i])
        last = lab_n[ki - 1] if ki > 0 else ""
        nxt = lab_n[ki] if ki < len(lab_n) else ""
        seg = POINTS.index(last) + 1 if last else 0
        prog = ""
        if last and nxt:
            prog = f"{(t - lab_t[ki - 1]) / (lab_t[ki] - lab_t[ki - 1]):.4f}"
        lines.append(f"{t:.2f},{fmt(*res['left'], i)},{fmt(*res['right'], i)},"
                     f"{label_col[i]},{last},{nxt},{seg},{prog}")
    (folder / "labeled.csv").write_text("\n".join(lines) + "\n")

    cover = {leg: 100.0 * res[leg][1].mean() for leg in LEGS}
    marks = " ".join(f"{n}@{t:.1f}s" for t, n in labels) or "no labels"
    missing = [p for p in POINTS if p not in lab_n]
    warn = f" | MISSING: {', '.join(missing)}" if missing else ""
    return (f"{folder.name}/labeled.csv: {len(grid) / GRID_HZ:.1f} s | left data {cover['left']:.0f}%, "
            f"right {cover['right']:.0f}% | {marks}{warn}")


# ----------------------------------------------------------------------------
# Receiver with labels
# ----------------------------------------------------------------------------
class LabelingReceiver(Receiver):
    def __init__(self, local_port=base.LOCAL_PORT):
        super().__init__(local_port)
        self.trial_name = "route"
        self.labels = []            # (t_s, point) of the current / last recording
        self.labels_file = None
        self.last_summary = ""

    def play(self):
        with self.lock:
            if self.state == "recording":
                return
            if self.state == "stopped":
                RECORD_DIR.mkdir(exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                name = re.sub(r"[^A-Za-z0-9_-]+", "_", self.trial_name).strip("_") or "route"
                self.folder = RECORD_DIR / f"{name}_{stamp}"
                self.folder.mkdir()
                for leg in self.legs.values():
                    leg.csv = open(self.folder / f"{leg.name}.csv", "w", buffering=1)
                    leg.csv.write(base.CSV_HEADER)
                self.labels = []
                self.labels_file = open(self.folder / "labels.csv", "w", buffering=1)
                self.labels_file.write("t_s,label,pc_time\n")
                self.rec_t0 = time.time() - self.t0
                self.last_summary = ""
            self.state = "recording"

    @property
    def next_point(self):
        """The point the next label must be (None once End is marked)."""
        if self.state == "stopped":
            return POINTS[0]
        n = len(self.labels)
        return POINTS[n] if n < len(POINTS) else None

    def mark(self, point):
        """Label the current instant. Returns an error string, or None on success."""
        if point != self.next_point:
            return f"Next label must be {POINT_NAMES.get(self.next_point, 'nothing (route complete)')}"
        if point == "start" and self.state != "recording":
            self.play()
        with self.lock:
            if self.state != "recording":
                return "Resume recording (Record) before labeling"
            now = time.time()
            t = now - self.t0 - self.rec_t0
            self.labels.append((t, point))
            self.labels_file.write(f"{t:.3f},{point},{now:.3f}\n")
        return None

    def undo(self):
        with self.lock:
            if self.state == "stopped" or not self.labels:
                return
            self.labels.pop()
            self.labels_file.seek(0)
            self.labels_file.truncate()
            self.labels_file.write("t_s,label,pc_time\n")
            for t, p in self.labels:
                self.labels_file.write(f"{t:.3f},{p},{self.t0 + self.rec_t0 + t:.3f}\n")

    def stop_recording(self):
        with self.lock:
            folder = self.folder if self.state != "stopped" else None
            self.state = "stopped"
            if self.labels_file:
                self.labels_file.close()
                self.labels_file = None
        super().stop_recording()              # closes left.csv / right.csv
        if folder:
            self.last_summary = build_labeled_csv(folder)
            print(self.last_summary)


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
def run_gui(rx: LabelingReceiver, autoconnect: dict):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    COLORS = {"left": "#1f77b4", "right": "#ff7f0e"}

    def load_settings():
        try:
            return base.json.loads(base.SETTINGS_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def save_settings(d):
        try:
            base.SETTINGS_FILE.write_text(base.json.dumps(d, indent=2))
        except OSError:
            pass

    def no_focus(btn):
        btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)   # Space must not re-click the last button
        return btn

    class LegPanel(QtWidgets.QGroupBox):
        def __init__(self, leg):
            super().__init__(f"{leg.capitalize()} leg")
            self.leg = leg
            self.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {COLORS[leg]}; }}")
            row = QtWidgets.QHBoxLayout(self)
            self.dot = QtWidgets.QLabel("●")
            row.addWidget(self.dot)
            row.addWidget(QtWidgets.QLabel("Pico IP:"))
            self.ip_box = QtWidgets.QLineEdit()
            self.ip_box.setPlaceholderText("printed by the Pico, e.g. 10.0.0.251")
            self.ip_box.setMinimumWidth(170)
            self.ip_box.returnPressed.connect(self.toggle)
            row.addWidget(self.ip_box)
            self.btn = no_focus(QtWidgets.QPushButton("Connect"))
            self.btn.clicked.connect(self.toggle)
            row.addWidget(self.btn)
            self.stats = QtWidgets.QLabel("Not connected")
            self.stats.setMinimumWidth(330)
            row.addWidget(self.stats, 1)
            self.last_count = (time.time(), 0)
            self.rate = 0.0

        def toggle(self):
            if rx.legs[self.leg].target is not None:
                rx.set_target(self.leg, None)
                self.btn.setText("Connect")
                self.ip_box.setEnabled(True)
                return
            self.connect_to(self.ip_box.text())

        def connect_to(self, text):
            try:
                rx.set_target(self.leg, parse_address(text))
            except ValueError as e:
                self.stats.setText(str(e))
                return
            self.ip_box.setText(text.strip())
            self.ip_box.setEnabled(False)
            self.btn.setText("Disconnect")
            self.last_count = (time.time(), 0)
            s = load_settings()
            s[self.leg] = text.strip()
            save_settings(s)

        def refresh(self):
            s = rx.legs[self.leg]
            now = time.time()
            if now - self.last_count[0] >= 1.0:
                self.rate = (s.samples - self.last_count[1]) / (now - self.last_count[0])
                self.last_count = (now, s.samples)
            if s.target is None:
                color, text = "#999999", "Not connected"
            elif s.connected:
                loss = 100.0 * s.lost / max(1, s.lost + s.packets)
                color = "#2ca02c" if loss < 5 else "#d6a000"
                text = f"{self.rate:.0f} samples/s | packets {s.packets} | lost {s.lost} ({loss:.1f}%)"
            elif s.packets:
                color, text = "#d62728", f"No data for {now - s.last_rx:.0f} s (Pico off or out of range?)"
            else:
                color, text = "#d6a000", f"Waiting for {s.target[0]}..."
            self.dot.setStyleSheet(f"color: {color}; font-size: 18px;")
            self.stats.setText(text)

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Two-leg IMU: route labeling")
            layout = QtWidgets.QVBoxLayout(self)
            st = load_settings()

            self.panels = {}
            for leg in LEGS:
                p = LegPanel(leg)
                p.ip_box.setText(st.get(leg, ""))
                self.panels[leg] = p
                layout.addWidget(p)

            # ---- recording row
            ctrl = QtWidgets.QHBoxLayout()
            ctrl.addWidget(QtWidgets.QLabel("Trial name:"))
            self.trial_box = QtWidgets.QLineEdit(st.get("trial", "route"))
            self.trial_box.setMaximumWidth(150)
            ctrl.addWidget(self.trial_box)
            self.play_btn = no_focus(QtWidgets.QPushButton("● Record"))
            self.pause_btn = no_focus(QtWidgets.QPushButton("⏸ Pause"))
            self.stop_btn = no_focus(QtWidgets.QPushButton("⏹ Stop"))
            self.play_btn.clicked.connect(self.start_recording)
            self.pause_btn.clicked.connect(rx.pause)
            self.stop_btn.clicked.connect(rx.stop_recording)
            for b in (self.play_btn, self.pause_btn, self.stop_btn):
                ctrl.addWidget(b)
            self.rec_label = QtWidgets.QLabel()
            ctrl.addWidget(self.rec_label, 1)
            ctrl.addWidget(QtWidgets.QLabel("Window:"))
            self.window_box = QtWidgets.QComboBox()
            for s in (5, 10, 30, 60, 120):
                self.window_box.addItem(f"{s} s", s)
            self.window_box.setCurrentIndex(2)
            ctrl.addWidget(self.window_box)
            self.show_box = {}
            for leg in LEGS:
                cb = QtWidgets.QCheckBox(leg.capitalize())
                cb.setChecked(True)
                cb.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
                cb.setStyleSheet(f"color: {COLORS[leg]}; font-weight: bold;")
                self.show_box[leg] = cb
                ctrl.addWidget(cb)
            layout.addLayout(ctrl)

            # ---- route labels
            box = QtWidgets.QGroupBox("Route labels")
            lab = QtWidgets.QVBoxLayout(box)
            row = QtWidgets.QHBoxLayout()
            self.point_btns = {}
            for p in POINTS:
                b = no_focus(QtWidgets.QPushButton(POINT_NAMES[p]))
                b.setMinimumHeight(36)
                b.clicked.connect(lambda _, p=p: self.mark(p))
                self.point_btns[p] = b
                row.addWidget(b)
            self.undo_btn = no_focus(QtWidgets.QPushButton("↶ Undo last label"))
            self.undo_btn.clicked.connect(rx.undo)
            row.addWidget(self.undo_btn)
            lab.addLayout(row)
            row2 = QtWidgets.QHBoxLayout()
            self.next_btn = no_focus(QtWidgets.QPushButton())
            self.next_btn.setMinimumHeight(48)
            f = self.next_btn.font()
            f.setPointSize(f.pointSize() + 5)
            f.setBold(True)
            self.next_btn.setFont(f)
            self.next_btn.clicked.connect(lambda: self.mark(rx.next_point))
            row2.addWidget(self.next_btn, 1)
            self.autostop = QtWidgets.QCheckBox("Auto-stop at End")
            self.autostop.setChecked(True)
            self.autostop.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            row2.addWidget(self.autostop)
            lab.addLayout(row2)
            self.label_status = QtWidgets.QLabel()
            self.label_status.setWordWrap(True)
            lab.addWidget(self.label_status)
            layout.addWidget(box)

            QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self,
                            activated=lambda: self.mark(rx.next_point))

            self.error_label = QtWidgets.QLabel()
            self.error_label.setStyleSheet("color: #d62728; font-weight: bold;")
            self.error_label.hide()
            layout.addWidget(self.error_label)

            # ---- plots
            self.plots = []
            for default in (CHANNELS.index("gz (dps)"), CHANNELS.index("|a| (g)")):
                cbox = QtWidgets.QComboBox()
                cbox.addItems(CHANNELS)
                cbox.setCurrentIndex(default)
                cbox.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
                plot = pg.PlotWidget()
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.setLabel("bottom", "time (s)")
                plot.addLegend(offset=(10, 10))
                curves = {leg: plot.plot(pen=pg.mkPen(COLORS[leg], width=2), name=leg.capitalize())
                          for leg in LEGS}
                cbox.currentIndexChanged.connect(lambda _, p=plot, b=cbox: p.setLabel("left", b.currentText()))
                plot.setLabel("left", cbox.currentText())
                if self.plots:
                    plot.setXLink(self.plots[0][1])
                layout.addWidget(cbox)
                layout.addWidget(plot, 1)
                self.plots.append((cbox, plot, curves, []))
            self.lines_key = None
            self.flash = ("", 0.0)

            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.refresh)
            self.timer.start(33)

        # ---- actions
        def missing_legs(self):
            return [l for l in LEGS if not rx.legs[l].connected]

        def confirm_legs(self):
            missing = self.missing_legs()
            if not missing:
                return True
            r = QtWidgets.QMessageBox.question(
                self, "Leg not streaming",
                f"No data from the {' and '.join(missing)} leg. Record anyway?")
            return r == QtWidgets.QMessageBox.StandardButton.Yes

        def start_recording(self):
            if rx.state == "stopped":
                if not self.confirm_legs():
                    return
                self.set_trial()
            rx.play()

        def set_trial(self):
            rx.trial_name = self.trial_box.text()
            s = load_settings()
            s["trial"] = self.trial_box.text()
            save_settings(s)

        def mark(self, point):
            if point is None:
                return
            if point == "start" and rx.state == "stopped":
                if not self.confirm_legs():
                    return
                self.set_trial()
            err = rx.mark(point)
            if err:
                self.flash = (err, time.time())
                return
            self.flash = (f"Marked {POINT_NAMES[point]}", time.time())
            if point == "end" and self.autostop.isChecked():
                folder = rx.folder
                QtCore.QTimer.singleShot(AUTO_STOP_DELAY_MS,
                                         lambda: rx.stop_recording() if rx.folder == folder else None)

        # ---- periodic update
        def update_label_lines(self):
            key = (rx.folder, len(rx.labels))
            if key == self.lines_key:
                return
            self.lines_key = key
            for _, plot, _, lines in self.plots:
                for ln in lines:
                    plot.removeItem(ln)
                lines.clear()
                if rx.rec_t0 is None:
                    continue
                for t, p in rx.labels:
                    ln = pg.InfiniteLine(pos=rx.rec_t0 + t, angle=90,
                                         pen=pg.mkPen("#2ca02c", width=2, style=QtCore.Qt.PenStyle.DashLine),
                                         label=POINT_NAMES[p],
                                         labelOpts={"position": 0.92, "color": "#2ca02c", "fill": (0, 0, 0, 160)})
                    plot.addItem(ln)
                    lines.append(ln)

        def refresh(self):
            if rx.error:
                self.error_label.setText(rx.error)
                self.error_label.show()
            for p in self.panels.values():
                p.refresh()

            w = self.window_box.currentData()
            end = rx.now() - 0.05
            snaps = {leg: rx.snapshot(leg, w, end) for leg in LEGS}
            for cbox, plot, curves, _ in self.plots:
                c = cbox.currentIndex()
                for leg in LEGS:
                    t, d = snaps[leg]
                    if self.show_box[leg].isChecked() and len(t):
                        curves[leg].setData(t, d[c])
                    else:
                        curves[leg].setData([], [])
                plot.setXRange(end - w, end, padding=0)
            self.update_label_lines()

            state = rx.state
            self.play_btn.setEnabled(state != "recording")
            self.play_btn.setText("● Resume" if state == "paused" else "● Record")
            self.pause_btn.setEnabled(state == "recording")
            self.stop_btn.setEnabled(state != "stopped")
            self.trial_box.setEnabled(state == "stopped")
            name = rx.folder.name if rx.folder else None
            if state == "recording":
                missing = self.missing_legs()
                warn = f"  <b style='color:#d62728'>(no data from {', '.join(missing)} leg!)</b>" if missing else ""
                self.rec_label.setText(f"<b style='color:#d62728'>● Recording</b> "
                                       f"{rx.now() - rx.rec_t0:.0f} s to {name}/{warn}")
            elif state == "paused":
                self.rec_label.setText(f"⏸ Paused ({name}/)")
            else:
                self.rec_label.setText("Not recording")

            nxt = rx.next_point
            can_mark = state == "recording" or nxt == "start"
            for p, b in self.point_btns.items():
                b.setEnabled(p == nxt and can_mark)
                b.setStyleSheet("QPushButton:enabled { background: #2ca02c; color: white; font-weight: bold; }"
                                if p == nxt and can_mark else "")
            self.undo_btn.setEnabled(state != "stopped" and bool(rx.labels))
            if nxt is None:
                self.next_btn.setText("Route complete ✓  (Stop to save)")
            elif state == "paused":
                self.next_btn.setText("Paused: press Resume before labeling")
            else:
                self.next_btn.setText(f"Mark {POINT_NAMES[nxt]}   [Space]"
                                      + ("  (starts recording)" if nxt == "start" and state == "stopped" else ""))
            self.next_btn.setEnabled(nxt is not None and can_mark)

            marks = "   ".join(f"<b>{POINT_NAMES[p]}</b> {t:.1f} s" for t, p in rx.labels) or "No labels yet"
            msg, at = self.flash
            flash = f"   <span style='color:#1f77b4'>{msg}</span>" if msg and time.time() - at < 3 else ""
            saved = (f"<br><span style='color:#555'>Saved: {rx.last_summary}</span>"
                     if rx.last_summary and state == "stopped" else "")
            self.label_status.setText(marks + flash + saved)

        def closeEvent(self, e):
            rx.stop_recording()
            super().closeEvent(e)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = Window()
    for leg, addr in autoconnect.items():
        if addr:
            win.panels[leg].connect_to(addr)
    win.resize(1200, 900)
    win.show()
    return app, win


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", help="left-leg Pico IP (optionally ip:port); connects on startup")
    ap.add_argument("--right", help="right-leg Pico IP (optionally ip:port); connects on startup")
    ap.add_argument("--port", type=int, default=base.LOCAL_PORT, help="local UDP port to listen on")
    ap.add_argument("--rebuild", type=Path, nargs="+", metavar="FOLDER",
                    help="regenerate labeled.csv for recording folder(s) and exit")
    args = ap.parse_args()

    if args.rebuild:
        for f in args.rebuild:
            print(build_labeled_csv(f))
        return

    rx = LabelingReceiver(args.port)
    rx.start()
    app, _win = run_gui(rx, {"left": args.left, "right": args.right})
    code = app.exec()
    rx.stop_flag = True
    rx.join(timeout=1.0)
    sys.exit(code)


if __name__ == "__main__":
    main()
