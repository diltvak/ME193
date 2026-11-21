"""Receive MPU6050 streams from TWO Pico Ws (left and right leg) at once, plot them live, and log to CSV.

    pip install -r requirements.txt
    python receiver.py                                    # type the two IPs into the GUI
    python receiver.py --left 10.0.0.251 --right 10.0.0.252   # connect on startup
    python receiver.py --left 127.0.0.1:5006 --right 127.0.0.1:5007   # with fake_pico.py

Both Picos run the same pico_main.py. The PC sends each one a "hello" every second from a single
UDP socket (port 5005). Each Pico streams back to that socket, and the stream is assigned to a leg by
the sender's address.

Time base: each Pico timestamps samples with its own microsecond counter. To line the two legs up,
each Pico's clock is mapped onto PC time with the smallest observed (PC receive time - Pico sample
time), i.e. the least-delayed packet. That makes the two legs agree to within a few ms, which
is enough for gait comparison (one stride is about 1 s).

Recording: Record starts a folder recordings/<timestamp>/ with left.csv and right.csv. Both files
use the old pc_receiver.py column format and share the same t_s = 0, so the step_nn / activity code
can read each leg directly.
"""
from __future__ import annotations

import argparse
import datetime
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETTINGS_FILE = HERE / "settings.json"     # remembers the last IPs typed in
RECORD_DIR = HERE / "recordings"

PICO_PORT = 5005             # port the Picos listen on (pico_main.py PORT)
LOCAL_PORT = 5005            # port this PC listens on
HELLO_PERIOD_S = 1.0         # Pico stops streaming after 5 s without a hello
CONNECTED_TIMEOUT_S = 1.5    # no packet for this long -> shown as not connected
ACCEL_LSB_PER_G = 8192.0     # +/-4 g, must match pico_main.py
GYRO_LSB_PER_DPS = 65.5      # +/-500 dps
TICKS_MOD = 1 << 30          # MicroPython ticks_us wraps at 2**30
MAX_SAMPLES = 30000          # per leg, ~5 min at 100 Hz kept for plotting
CLOCK_DRIFT_ALLOW = 2e-4     # s/s the clock offset may creep up (crystal drift between Pico and PC)

LEGS = ("left", "right")
# Channel rows held in memory. The first 7 are what the CSV stores (same order as pc_receiver.py).
CHANNELS = ["ax (g)", "ay (g)", "az (g)", "gx (dps)", "gy (dps)", "gz (dps)", "temp (C)",
            "|a| (g)", "|gyro| (dps)"]
N_CH = len(CHANNELS)
CSV_HEADER = "t_s,pc_time,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_c\n"

REC = np.dtype([("t", "<u4"), ("v", ">i2", (7,))])   # raw order: ax ay az temp gx gy gz
HEADER = struct.Struct("<IB")


def parse_address(text):
    """'10.0.0.5' or '10.0.0.5:5006' -> (ip, port). Raises ValueError."""
    text = text.strip()
    ip, _, port = text.partition(":")
    try:
        socket.inet_aton(ip)
    except OSError:
        raise ValueError(f"invalid IP address: {ip!r}")
    if ip.count(".") != 3:
        raise ValueError(f"invalid IP address: {ip!r}")
    port = int(port) if port else PICO_PORT
    return ip, port


class LegStream:
    """Everything about one Pico's stream: packet bookkeeping, clock mapping, buffers, CSV file."""

    def __init__(self, name):
        self.name = name
        self.target = None          # (ip, port) we send hellos to and accept packets from
        self.csv = None
        self.reset()

    def reset(self):
        """Forget the stream (new target, or Pico rebooted)."""
        self.last_seq = None
        self.last_raw = None
        self.abs_us = 0.0           # unwrapped Pico time of the last sample
        self.offset = None          # PC time - Pico time (s), from the least-delayed packet
        self.offset_at = 0.0        # PC time when offset was last updated
        self.t = np.empty(0)        # shared time base (s since the receiver started)
        self.data = np.empty((N_CH, 0))
        self.packets = 0
        self.samples = 0
        self.lost = 0
        self.last_rx = 0.0

    @property
    def connected(self):
        return self.target is not None and time.time() - self.last_rx < CONNECTED_TIMEOUT_S


class Receiver(threading.Thread):
    """One UDP socket serving both Picos. Thread-safe: the GUI reads through snapshot()/status()."""

    def __init__(self, local_port=LOCAL_PORT):
        super().__init__(daemon=True)
        self.local_port = local_port
        self.lock = threading.Lock()
        self.legs = {name: LegStream(name) for name in LEGS}
        self.t0 = time.time()       # zero of the shared time base
        self.stop_flag = False
        self.unknown_packets = 0
        self.state = "stopped"      # stopped | recording | paused
        self.folder = None
        self.rec_t0 = None          # shared-time value that becomes t_s = 0 in the CSVs
        self.error = None

    # ---- control (called from the GUI thread) ---------------------------------
    def set_target(self, leg, addr):
        """Point a leg at a Pico (ip, port), or None to disconnect."""
        with self.lock:
            other = self.legs["right" if leg == "left" else "left"]
            if addr is not None and other.target == addr:
                raise ValueError(f"{other.name} leg is already using {addr[0]}:{addr[1]}")
            self.legs[leg].target = addr
            self.legs[leg].reset()

    def play(self):
        """Start a new recording folder, or resume a paused one."""
        with self.lock:
            if self.state == "recording":
                return
            if self.state == "stopped":
                RECORD_DIR.mkdir(exist_ok=True)
                self.folder = RECORD_DIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.folder.mkdir()
                for leg in self.legs.values():
                    leg.csv = open(self.folder / f"{leg.name}.csv", "w", buffering=1)
                    leg.csv.write(CSV_HEADER)
                self.rec_t0 = time.time() - self.t0
            self.state = "recording"

    def pause(self):
        with self.lock:
            if self.state == "recording":
                self.state = "paused"

    def stop_recording(self):
        with self.lock:
            self.state = "stopped"
            for leg in self.legs.values():
                if leg.csv:
                    leg.csv.close()
                    leg.csv = None

    # ---- network thread -------------------------------------------------------
    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self.local_port))
        except OSError as e:
            self.error = f"Cannot open UDP port {self.local_port}: {e} (is pc_receiver.py still running?)"
            print(self.error)
            return
        sock.settimeout(0.1)
        last_hello = 0.0
        while not self.stop_flag:
            if time.time() - last_hello >= HELLO_PERIOD_S:
                last_hello = time.time()
                for leg in self.legs.values():
                    target = leg.target
                    if target:
                        try:
                            sock.sendto(b"hello", target)
                        except OSError as e:
                            print(f"hello to {leg.name} ({target[0]}) failed: {e}")
            try:
                pkt, sender = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                continue
            leg = next((l for l in self.legs.values() if l.target == sender), None)
            if leg is None:
                self.unknown_packets += 1
                continue
            try:
                self.handle(leg, pkt)
            except Exception as e:  # malformed packet
                print(f"bad packet from {leg.name}: {e}")
        sock.close()
        self.stop_recording()

    def handle(self, leg: LegStream, pkt):
        seq, n = HEADER.unpack_from(pkt)
        if n == 0 or len(pkt) != HEADER.size + n * REC.itemsize:
            return
        rx_time = time.time()
        rec = np.frombuffer(pkt, dtype=REC, count=n, offset=HEADER.size)

        with self.lock:
            if leg.last_seq is not None and seq == 0:          # Pico rebooted: restart its time base
                leg.last_seq = leg.last_raw = leg.offset = None
                leg.abs_us = 0.0
            if leg.last_seq is not None:
                if seq <= leg.last_seq:                         # duplicate / reordered
                    return
                leg.lost += seq - leg.last_seq - 1
            leg.last_seq = seq

            raw = rec["t"].astype(np.int64)
            prev = raw[0] if leg.last_raw is None else leg.last_raw
            steps = np.diff(raw, prepend=prev) % TICKS_MOD       # unwrap 30-bit counter
            pico_s = (leg.abs_us + np.cumsum(steps)) / 1e6
            leg.abs_us = pico_s[-1] * 1e6
            leg.last_raw = raw[-1]

            # Map Pico time onto PC time. The last sample of a packet was taken just before it was
            # sent, so rx_time - pico_s[-1] = clock offset + network delay. The smallest value seen
            # has the least delay. Let it creep up slowly so crystal drift cannot pin it forever.
            est = rx_time - pico_s[-1]
            if leg.offset is None:
                leg.offset = est
            else:
                leg.offset = min(leg.offset + CLOCK_DRIFT_ALLOW * (rx_time - leg.offset_at), est)
            leg.offset_at = rx_time
            t = pico_s + leg.offset - self.t0

            v = rec["v"].astype(np.float64)
            a = v[:, 0:3] / ACCEL_LSB_PER_G
            g = v[:, 4:7] / GYRO_LSB_PER_DPS
            d = np.vstack([a.T, g.T, v[:, 3] / 340.0 + 36.53,
                           np.linalg.norm(a, axis=1), np.linalg.norm(g, axis=1)])

            if self.state == "recording" and leg.csv:
                keep = t >= self.rec_t0
                leg.csv.write("".join(
                    f"{t[i] - self.rec_t0:.4f},{rx_time:.3f}," + ",".join(f"{d[c, i]:.5f}" for c in range(7)) + "\n"
                    for i in np.flatnonzero(keep)))
            leg.t = np.concatenate([leg.t, t])[-MAX_SAMPLES:]
            leg.data = np.concatenate([leg.data, d], axis=1)[:, -MAX_SAMPLES:]
            leg.packets += 1
            leg.samples += n
            leg.last_rx = rx_time

    # ---- reads for the GUI ----------------------------------------------------
    def now(self):
        return time.time() - self.t0

    def snapshot(self, leg, window_s, end):
        """(t, data) for one leg covering [end - window_s, end]."""
        with self.lock:
            s = self.legs[leg]
            i = np.searchsorted(s.t, end - window_s - 0.5)
            return s.t[i:].copy(), s.data[:, i:].copy()


# ============================================================================
# GUI
# ============================================================================
def run_gui(rx: Receiver, autoconnect: dict):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    COLORS = {"left": "#1f77b4", "right": "#ff7f0e"}

    def load_settings():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def save_settings(d):
        try:
            SETTINGS_FILE.write_text(json.dumps(d, indent=2))
        except OSError:
            pass

    class LegPanel(QtWidgets.QGroupBox):
        """IP entry, connect button and live stats for one leg."""

        def __init__(self, leg, window):
            super().__init__(f"{leg.capitalize()} leg")
            self.leg, self.window = leg, window
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
            self.btn = QtWidgets.QPushButton("Connect")
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
                addr = parse_address(text)
                rx.set_target(self.leg, addr)
            except ValueError as e:
                self.stats.setText(str(e))
                return
            self.ip_box.setText(text.strip())
            self.ip_box.setEnabled(False)
            self.btn.setText("Disconnect")
            self.last_count = (time.time(), 0)
            self.rate = 0.0
            s = load_settings()
            s[self.leg] = text.strip()
            save_settings(s)

        def refresh(self):
            s = rx.legs[self.leg]
            now = time.time()
            dt = now - self.last_count[0]
            if dt >= 1.0:
                self.rate = (s.samples - self.last_count[1]) / dt
                self.last_count = (now, s.samples)
            if s.target is None:
                color, text = "#999999", "Not connected"
            elif s.connected:
                loss = 100.0 * s.lost / max(1, s.lost + s.packets)
                color = "#2ca02c" if loss < 5 else "#d6a000"
                text = (f"{self.rate:.0f} samples/s | packets {s.packets} | lost {s.lost} ({loss:.1f}%)")
            elif s.packets:
                color, text = "#d62728", f"No data for {now - s.last_rx:.0f} s (Pico off or out of range?)"
            else:
                color, text = "#d6a000", f"Waiting for {s.target[0]}..."
            self.dot.setStyleSheet(f"color: {color}; font-size: 18px;")
            self.stats.setText(text)

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Two-leg IMU live")
            layout = QtWidgets.QVBoxLayout(self)

            settings = load_settings()
            self.panels = {}
            for leg in LEGS:
                p = LegPanel(leg, self)
                p.ip_box.setText(settings.get(leg, ""))
                self.panels[leg] = p
                layout.addWidget(p)

            ctrl = QtWidgets.QHBoxLayout()
            self.play_btn = QtWidgets.QPushButton("● Record")
            self.pause_btn = QtWidgets.QPushButton("⏸ Pause")
            self.stop_btn = QtWidgets.QPushButton("⏹ Stop")
            self.play_btn.clicked.connect(rx.play)
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
            self.window_box.setCurrentIndex(1)
            ctrl.addWidget(self.window_box)
            ctrl.addWidget(QtWidgets.QLabel("Show:"))
            self.show_box = {}
            for leg in LEGS:
                cb = QtWidgets.QCheckBox(leg.capitalize())
                cb.setChecked(True)
                cb.setStyleSheet(f"color: {COLORS[leg]}; font-weight: bold;")
                self.show_box[leg] = cb
                ctrl.addWidget(cb)
            self.pause_plot = QtWidgets.QCheckBox("Freeze plot")
            ctrl.addWidget(self.pause_plot)
            layout.addLayout(ctrl)

            self.error_label = QtWidgets.QLabel()
            self.error_label.setStyleSheet("color: #d62728; font-weight: bold;")
            self.error_label.hide()
            layout.addWidget(self.error_label)

            self.plots = []
            for default in (CHANNELS.index("gz (dps)"), CHANNELS.index("|a| (g)")):
                box = QtWidgets.QComboBox()
                box.addItems(CHANNELS)
                box.setCurrentIndex(default)
                plot = pg.PlotWidget()
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.setLabel("bottom", "time (s)")
                plot.addLegend(offset=(10, 10))
                curves = {leg: plot.plot(pen=pg.mkPen(COLORS[leg], width=2), name=leg.capitalize())
                          for leg in LEGS}
                box.currentIndexChanged.connect(lambda _, p=plot, b=box: p.setLabel("left", b.currentText()))
                plot.setLabel("left", box.currentText())
                if self.plots:
                    plot.setXLink(self.plots[0][1])
                layout.addWidget(box)
                layout.addWidget(plot, 1)
                self.plots.append((box, plot, curves))

            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.refresh)
            self.timer.start(33)

        def refresh(self):
            if rx.error:
                self.error_label.setText(rx.error)
                self.error_label.show()
            for p in self.panels.values():
                p.refresh()

            if not self.pause_plot.isChecked():
                w = self.window_box.currentData()
                end = rx.now() - 0.05       # ~one packet of latency, so the right edge is not empty
                snaps = {leg: rx.snapshot(leg, w, end) for leg in LEGS}
                for box, plot, curves in self.plots:
                    c = box.currentIndex()
                    for leg in LEGS:
                        t, d = snaps[leg]
                        if self.show_box[leg].isChecked() and len(t):
                            curves[leg].setData(t, d[c])
                        else:
                            curves[leg].setData([], [])
                    plot.setXRange(end - w, end, padding=0)

            state = rx.state
            self.play_btn.setEnabled(state != "recording")
            self.pause_btn.setEnabled(state == "recording")
            self.stop_btn.setEnabled(state != "stopped")
            name = f"{rx.folder.parent.name}/{rx.folder.name}" if rx.folder else None
            if state == "recording":
                missing = [l for l in LEGS if not rx.legs[l].connected]
                warn = f"  (no data from {', '.join(missing)} leg!)" if missing else ""
                self.rec_label.setText(f"<b style='color:#d62728'>● Recording</b> to {name}/{warn}")
            elif state == "paused":
                self.rec_label.setText(f"⏸ Paused ({name}/)")
            else:
                self.rec_label.setText("Not recording" + (f" (last: {name}/)" if name else ""))

        def closeEvent(self, e):
            rx.stop_recording()
            super().closeEvent(e)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = Window()
    for leg, addr in autoconnect.items():
        if addr:
            win.panels[leg].connect_to(addr)
    win.resize(1150, 800)
    win.show()
    return app, win


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", help="left-leg Pico IP (optionally ip:port); connects on startup")
    ap.add_argument("--right", help="right-leg Pico IP (optionally ip:port); connects on startup")
    ap.add_argument("--port", type=int, default=LOCAL_PORT, help="local UDP port to listen on")
    args = ap.parse_args()

    rx = Receiver(args.port)
    rx.start()
    app, _win = run_gui(rx, {"left": args.left, "right": args.right})
    code = app.exec()
    rx.stop_flag = True
    rx.join(timeout=1.0)
    sys.exit(code)


if __name__ == "__main__":
    main()
