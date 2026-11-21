"""Walking directions to the destination from the two leg IMUs. Live GUI.

    python3 -m navigator.navigate --left 10.247.137.121 --right 10.247.137.102 --height 1.68 --voice
    python3 -m navigator.navigate                                   # type IPs / height in the window

    # no hardware: replay a recorded walk, optionally disguised as "another person"
    python3 -m navigator.navigate --replay recordings/training3_20261120_123004 --speed 2
    python3 -m navigator.navigate --replay recordings/training3_20261120_123004 --other-person --height 1.55

How to use it with a new walker:
  1. Strap a Pico on each shin (any orientation, either leg). Connect both.
  2. Enter the walker's height (used only for the first guess of their step length).
  3. Have them stand still at the start point for ~2 s (gyro calibration; the Start button turns green).
  4. Press Space / Start and let them walk. Directions update continuously; with --voice they are spoken.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C
from .core import Navigator, load_route

sys.path.insert(0, str(C.ROOT))
SETTINGS_FILE = C.ROOT / "settings.json"


# ============================================================================
# data sources: live Picos or a replayed recording
# ============================================================================
class LiveSource:
    """Wraps receiver.Receiver: hands out samples that arrived since the last call."""

    def __init__(self, port):
        import receiver
        self.mod = receiver
        self.rx = receiver.Receiver(port)
        self.rx.start()
        self.last_t = {leg: -1e9 for leg in C.LEGS}
        self.auto_start = None

    def now(self):
        return self.rx.now()

    def new_samples(self, leg):
        with self.rx.lock:
            s = self.rx.legs[leg]
            i = int(np.searchsorted(s.t, self.last_t[leg], side="right"))
            t, d = s.t[i:].copy(), s.data[:, i:].copy()
        if len(t):
            self.last_t[leg] = t[-1]
        return t, d[0:3].T, d[3:6].T

    def status(self, leg):
        s = self.rx.legs[leg]
        if s.target is None:
            return "#999999", "not connected"
        if s.connected:
            loss = 100.0 * s.lost / max(1, s.lost + s.packets)
            return ("#2ca02c" if loss < 5 else "#d6a000"), f"streaming, {loss:.0f}% packets lost"
        if s.packets:
            return "#d62728", f"no data for {time.time() - s.last_rx:.0f} s"
        return "#d6a000", f"waiting for {s.target[0]}..."

    def connect(self, leg, text):
        addr = None if not text else self.mod.parse_address(text)
        self.rx.set_target(leg, addr)
        self.last_t[leg] = -1e9

    def close(self):
        self.rx.stop_flag = True
        self.rx.join(timeout=1.0)


class ReplaySource:
    """Plays a recorded walk at wall-clock speed (x speed), optionally disguised as another person."""

    def __init__(self, folder, speed=1.0, other_person=False, seed=0):
        from .data import load_walk
        from .evaluate import augment
        walk = load_walk(Path(folder))
        rng = np.random.default_rng(seed)
        if other_person:
            self.warp = float(rng.choice([0.82, 0.85, 1.15, 1.2]))
            self.legs, self.labels = augment(walk, rng, rotate=True, swap=True, warp=self.warp,
                                             loss=0.10, dead_zone=6.0)
        else:
            self.warp = 1.0
            self.legs, self.labels = augment(walk, rng)
        self.name = walk.name + (f" (as another person: sensors rotated, legs swapped, "
                                 f"{'slower' if self.warp > 1 else 'faster'} x{1 / self.warp:.2f})"
                                 if other_person else "")
        self.speed = speed
        self.t_first = min(float(v[0][0]) for v in self.legs.values() if len(v[0]))
        self.t_last = max(float(v[0][-1]) for v in self.legs.values() if len(v[0]))
        self.wall0 = time.time()
        self.idx = {leg: 0 for leg in C.LEGS}
        self.auto_start = self.labels["start"]          # press Start for the walker at the labelled moment

    def now(self):
        return min(self.t_first + (time.time() - self.wall0) * self.speed, self.t_last + 5.0)

    def new_samples(self, leg):
        t, a, g = self.legs[leg]
        i0 = self.idx[leg]
        i1 = int(np.searchsorted(t, self.now(), side="right"))
        self.idx[leg] = i1
        return t[i0:i1], a[i0:i1], g[i0:i1]

    def status(self, leg):
        t = self.legs[leg][0]
        if not len(t):
            return "#999999", "no data in file"
        return "#2ca02c", "replay"

    def connect(self, leg, text):
        pass

    def close(self):
        pass


# ============================================================================
# route drawing
# ============================================================================
def route_geometry(route):
    """Node (x, y) in meters: start at the origin facing up; + turn = left (counter-clockwise)."""
    heading = np.pi / 2
    xy = [(0.0, 0.0)]
    nodes = route["nodes"]
    for a, b in zip(nodes, nodes[1:]):
        heading += np.radians(a["turn_deg"])
        d = b["dist_m"] - a["dist_m"]
        x, y = xy[-1]
        xy.append((x + d * np.cos(heading), y + d * np.sin(heading)))
    return np.array(xy), np.array([n["dist_m"] for n in nodes])


def point_at(s, xy, dist):
    s = np.clip(s, dist[0], dist[-1])
    return np.interp(s, dist, xy[:, 0]), np.interp(s, dist, xy[:, 1])


# ============================================================================
# GUI
# ============================================================================
STATE_COLORS = {"walking": "#1f6fb4", "turn_now": "#d97706", "arrived": "#15803d",
                "off_route": "#b91c1c", "passed_turn": "#b91c1c", "calibrate": "#6b7280", "ready": "#6b7280"}


def run(source, route, height, voice):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    def load_settings():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def save_settings(**kw):
        s = load_settings()
        s.update(kw)
        try:
            SETTINGS_FILE.write_text(json.dumps(s, indent=2))
        except OSError:
            pass

    say_cmd = shutil.which("say")

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Two-leg IMU navigator")
            self.resize(1000, 820)
            self.nav = None
            self.spoken = None
            self.voice_proc = None
            self.announced_near = set()
            self.n_msgs = 0
            self.n_events = 0
            settings = load_settings()
            lay = QtWidgets.QVBoxLayout(self)

            # --- legs
            self.leg_rows = {}
            for leg in C.LEGS:
                row = QtWidgets.QHBoxLayout()
                dot = QtWidgets.QLabel("●")
                row.addWidget(dot)
                row.addWidget(QtWidgets.QLabel(f"<b>{leg.capitalize()} leg</b>"))
                ip = QtWidgets.QLineEdit(settings.get(leg, ""))
                ip.setPlaceholderText("Pico IP")
                ip.setMaximumWidth(170)
                btn = QtWidgets.QPushButton("Connect")
                btn.clicked.connect(lambda _, l=leg: self.toggle_leg(l))
                ip.returnPressed.connect(lambda l=leg: self.toggle_leg(l))
                if isinstance(source, ReplaySource):
                    ip.setText(source.name if leg == "left" else "")
                    ip.setPlaceholderText("")
                    ip.setEnabled(False)
                    ip.setMaximumWidth(10000)
                    btn.hide()
                row.addWidget(ip, 1 if isinstance(source, ReplaySource) else 0)
                row.addWidget(btn)
                stat = QtWidgets.QLabel()
                row.addWidget(stat, 1)
                lay.addLayout(row)
                self.leg_rows[leg] = (dot, ip, btn, stat)

            # --- walker + start
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Walker's height (m):"))
            self.height = QtWidgets.QDoubleSpinBox()
            self.height.setRange(1.0, 2.3)
            self.height.setSingleStep(0.01)
            self.height.setValue(height or settings.get("nav_height", C.DEFAULT_TRAINER_HEIGHT))
            row.addWidget(self.height)
            self.start_btn = QtWidgets.QPushButton("Start (Space)")
            self.start_btn.clicked.connect(self.start)
            self.start_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            row.addWidget(self.start_btn)
            self.reset_btn = QtWidgets.QPushButton("Reset")
            self.reset_btn.clicked.connect(self.reset)
            self.reset_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            row.addWidget(self.reset_btn)
            self.voice_box = QtWidgets.QCheckBox("Speak directions")
            self.voice_box.setChecked(bool(voice and say_cmd))
            self.voice_box.setEnabled(bool(say_cmd))
            if not say_cmd:
                self.voice_box.setToolTip("needs the macOS 'say' command")
            row.addWidget(self.voice_box)
            row.addStretch(1)
            lay.addLayout(row)

            # --- instruction
            self.instr = QtWidgets.QLabel()
            self.instr.setWordWrap(True)
            self.instr.setMinimumHeight(110)
            self.instr.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(self.instr)
            self.info = QtWidgets.QLabel()
            self.info.setStyleSheet("font-size: 14px;")
            lay.addWidget(self.info)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setRange(0, 1000)
            self.progress.setTextVisible(True)
            lay.addWidget(self.progress)

            # --- map + log
            split = QtWidgets.QHBoxLayout()
            self.plot = pg.PlotWidget()
            self.plot.setAspectLocked(True)
            self.plot.getViewBox().setDefaultPadding(0.12)
            self.plot.showGrid(x=True, y=True, alpha=0.2)
            self.plot.setLabel("bottom", "m")
            self.plot.setLabel("left", "m")
            self.xy, self.dist = route_geometry(route)
            self.plot.plot(self.xy[:, 0], self.xy[:, 1], pen=pg.mkPen("#888", width=6))
            for (x, y), n in zip(self.xy, route["nodes"]):
                lab = n["name"] + ("" if not n["turn_deg"] else f" ({'L' if n['turn_deg'] > 0 else 'R'})")
                t = pg.TextItem(lab, color="#ddd", anchor=(0, 1))
                t.setPos(x + 0.8, y + 0.8)
                self.plot.addItem(t)
            self.plot.plot(self.xy[:, 0], self.xy[:, 1], pen=None, symbol="o", symbolSize=10,
                           symbolBrush="#ddd")
            self.cloud = self.plot.plot([], [], pen=None, symbol="o", symbolSize=4,
                                        symbolBrush=pg.mkBrush(80, 160, 255, 60), symbolPen=None)
            self.me = self.plot.plot([], [], pen=None, symbol="o", symbolSize=18, symbolBrush="#f59e0b")
            split.addWidget(self.plot, 3)
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(500)
            split.addWidget(self.log, 2)
            lay.addLayout(split, 1)

            QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.start)
            self.new_nav()
            if not isinstance(source, ReplaySource):
                for leg in C.LEGS:
                    if autoconnect.get(leg):
                        self.leg_rows[leg][1].setText(autoconnect[leg])
                        self.toggle_leg(leg)
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.refresh)
            self.timer.start(100)

        # ----------------------------------------------------------------------------
        def new_nav(self):
            self.nav = Navigator(route, self.height.value())
            self.spoken = None
            self.announced_near = set()
            self.n_msgs = self.n_events = 0

        def toggle_leg(self, leg):
            dot, ip, btn, stat = self.leg_rows[leg]
            try:
                if btn.text() == "Disconnect":
                    source.connect(leg, None)
                    btn.setText("Connect")
                    ip.setEnabled(True)
                else:
                    source.connect(leg, ip.text())
                    btn.setText("Disconnect")
                    ip.setEnabled(False)
                    save_settings(**{leg: ip.text().strip()})
            except ValueError as e:
                stat.setText(str(e))

        def start(self):
            if self.nav.started:
                return
            if not any(self.nav.calibrated().values()):
                self.write_log("Not calibrated yet: the walker must stand still for 2 s first.")
                return
            save_settings(nav_height=self.height.value())
            self.nav.step_len0 = C.STEP_LENGTH_PER_HEIGHT * self.height.value()
            self.nav.start(source.now())
            self.height.setEnabled(False)
            self.write_log(f"Started. Height {self.height.value():.2f} m -> first guess of step length "
                           f"{self.nav.step_len0:.2f} m.")

        def reset(self):
            old = self.nav.fusion
            self.new_nav()
            self.nav.fusion.legs = old.legs                 # keep the calibrated sensor state
            self.height.setEnabled(True)
            self.write_log("Reset. Go back to the start point and press Start.")

        def write_log(self, text):
            self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

        def speak(self, text):
            if not (self.voice_box.isChecked() and say_cmd):
                return
            if self.voice_proc and self.voice_proc.poll() is None:
                self.voice_proc.terminate()                 # newest instruction wins
            self.voice_proc = subprocess.Popen([say_cmd, text])

        # ----------------------------------------------------------------------------
        def refresh(self):
            for leg in C.LEGS:
                t, a, g = source.new_samples(leg)
                if len(t):
                    self.nav.feed(leg, t, a, g)
                col, txt = source.status(leg)
                lp = self.nav.fusion.legs[leg]
                if col == "#2ca02c" or len(lp.t):
                    txt += " | calibrated" if lp.bias_ready else " | stand still to calibrate"
                self.leg_rows[leg][0].setStyleSheet(f"color: {col}; font-size: 18px;")
                self.leg_rows[leg][3].setText(txt)

            if not self.nav.started and source.auto_start is not None and source.now() >= source.auto_start:
                self.start()
                source.auto_start = None
            cal = any(self.nav.calibrated().values())
            self.start_btn.setEnabled(cal and not self.nav.started)
            self.start_btn.setStyleSheet("background: #2ca02c; color: white; font-weight: bold;"
                                         if cal and not self.nav.started else "")

            g = self.nav.update(source.now())
            if g is None:
                state = "ready" if cal else "calibrate"
                text = ("Stand at the start point and press Start." if cal
                        else "Stand still for 2 seconds so the sensors can calibrate.")
                self.show_instruction(state, text)
                self.info.setText("")
                self.progress.setValue(0)
                self.progress.setFormat("")
                self.cloud.setData([], [])
                self.me.setData([], [])
                return

            self.show_instruction(g.state, g.text)
            nodes = route["nodes"]
            self.info.setText(
                f"Position {g.s:.1f} ± {g.s_sd:.1f} m of {self.nav.tracker.D:.0f} m  |  "
                f"turns done {g.turns_taken}/{self.nav.tracker.n_turns}  |  "
                f"step length {g.step_len:.2f} ± {g.step_len_sd:.2f} m  |  "
                f"next: {nodes[g.next_node]['name']} in {max(g.to_next_m, 0):.0f} m")
            frac = min(max(g.s / self.nav.tracker.D, 0), 1)
            self.progress.setValue(int(frac * 1000))
            self.progress.setFormat(f"{frac * 100:.0f}% of the way, {g.to_end_m:.0f} m to go")

            tr = self.nav.tracker
            pick = np.random.default_rng(0).choice(tr.N, 300, p=tr.w)
            x, y = point_at(tr.s[pick], self.xy, self.dist)
            self.cloud.setData(x, y)
            x, y = point_at(np.array([g.s]), self.xy, self.dist)
            self.me.setData(x, y)

            # log + voice
            for ev in tr.events[self.n_events:]:
                self.write_log(ev.strip())
            self.n_events = len(tr.events)
            for _, m in self.nav.messages[self.n_msgs:]:
                self.write_log(m)
                self.speak(m)
            self.n_msgs = len(self.nav.messages)
            key = (g.state, g.next_node, g.turns_taken)
            if key != self.spoken:
                self.spoken = key
                self.speak(g.text.replace("~", "about "))
            elif g.state == "walking" and g.to_next_m < 8 and g.next_node not in self.announced_near \
                    and g.turns_taken < tr.n_turns:
                self.announced_near.add(g.next_node)
                turn = nodes[g.next_node]["turn_deg"]
                note = nodes[g.next_node].get("note")
                self.speak(f"In {g.to_next_m:.0f} meters, turn {'left' if turn > 0 else 'right'}"
                           + (f" and {note}." if note else "."))

        def show_instruction(self, state, text):
            col = STATE_COLORS.get(state, "#1f6fb4")
            self.instr.setStyleSheet(f"font-size: 30px; font-weight: bold; color: white; background: {col};"
                                     f"border-radius: 10px; padding: 12px;")
            self.instr.setText(text)

        def closeEvent(self, e):
            source.close()
            e.accept()

    autoconnect = getattr(source, "autoconnect", {})
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    return app.exec()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", help="left-leg Pico IP (optionally ip:port)")
    ap.add_argument("--right", help="right-leg Pico IP (optionally ip:port)")
    ap.add_argument("--port", type=int, default=5005, help="local UDP port")
    ap.add_argument("--height", type=float, help="walker's height in m (first guess of their step length)")
    ap.add_argument("--route", default=str(C.ROUTE_FILE), help="route map from navigator.build_route")
    ap.add_argument("--voice", action="store_true", help="speak the directions (macOS 'say')")
    ap.add_argument("--replay", help="recordings/<folder> to replay instead of live Picos")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed")
    ap.add_argument("--other-person", action="store_true",
                    help="replay disguised as another walker (rotated sensors, swapped legs, other pace, packet loss)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not Path(args.route).exists():
        raise SystemExit(f"No route map at {args.route}. Run: python3 -m navigator.build_route")
    route = load_route(args.route)
    if args.replay:
        source = ReplaySource(args.replay, args.speed, args.other_person, args.seed)
    else:
        source = LiveSource(args.port)
        source.autoconnect = {"left": args.left, "right": args.right}
    sys.exit(run(source, route, args.height, args.voice))


if __name__ == "__main__":
    main()
