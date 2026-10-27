"""Receive MPU6050 UDP batches from the Pico, plot live, and log to CSV.

pip install numpy pyqtgraph PyQt6
python pc_receiver.py
"""
import datetime
import socket
import struct
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

PORT = 5005
ACCEL_LSB_PER_G = 8192.0     # +/-4 g, must match pico_main.py
GYRO_LSB_PER_DPS = 65.5      # +/-500 dps
TICKS_MOD = 1 << 30          # MicroPython ticks_us wraps at 2**30
MAX_SAMPLES = 30000          # ~5 min kept in memory for plotting

CHANNELS = ["ax (g)", "ay (g)", "az (g)", "gx (dps)", "gy (dps)", "gz (dps)", "temp (C)"]
REC = np.dtype([("t", "<u4"), ("v", ">i2", (7,))])   # raw order: ax ay az temp gx gy gz
HEADER = struct.Struct("<IB")


class Receiver(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.t = np.empty(0)
        self.data = np.empty((7, 0))
        self.lost = 0
        self.packets = 0
        self.samples = 0
        self.last_seq = None
        self.last_raw = None
        self.abs_us = 0.0
        self.t0_us = None
        self.stop = False
        self.target = None          # Pico (ip, port) once the user clicks Connect
        name = datetime.datetime.now().strftime("mpu_%Y%m%d_%H%M%S.csv")
        self.filename = name
        self.csv = open(name, "w", buffering=1)
        self.csv.write("t_s,pc_time,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_c\n")

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", PORT))
        sock.settimeout(0.5)
        last_hello = 0.0
        while not self.stop:
            # Hello packets tell the Pico where to stream; repeating them keeps it alive.
            if self.target and time.time() - last_hello >= 1.0:
                try:
                    sock.sendto(b"hello", self.target)
                except OSError as e:
                    print("hello failed:", e)
                last_hello = time.time()
            try:
                pkt, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            try:
                self.handle(pkt)
            except Exception as e:  # malformed packet
                print("bad packet:", e)
        self.csv.close()

    def handle(self, pkt):
        seq, n = HEADER.unpack_from(pkt)
        if len(pkt) != HEADER.size + n * REC.itemsize or n == 0:
            return
        rx_time = time.time()
        rec = np.frombuffer(pkt, dtype=REC, count=n, offset=HEADER.size)

        if self.last_seq is not None and seq == 0:       # Pico rebooted: restart time base
            self.last_raw = None
            self.last_seq = None
        if self.last_seq is not None:
            if seq <= self.last_seq:                      # duplicate / reordered
                return
            self.lost += seq - self.last_seq - 1
        self.last_seq = seq

        raw = rec["t"].astype(np.int64)
        prev = raw[0] if self.last_raw is None else self.last_raw
        steps = np.diff(raw, prepend=prev) % TICKS_MOD    # unwrap 30-bit counter
        times_us = self.abs_us + np.cumsum(steps)
        self.abs_us = times_us[-1]
        self.last_raw = raw[-1]
        if self.t0_us is None:
            self.t0_us = times_us[0]
        t = (times_us - self.t0_us) / 1e6

        v = rec["v"].astype(np.float64)
        d = np.vstack([
            v[:, 0] / ACCEL_LSB_PER_G, v[:, 1] / ACCEL_LSB_PER_G, v[:, 2] / ACCEL_LSB_PER_G,
            v[:, 4] / GYRO_LSB_PER_DPS, v[:, 5] / GYRO_LSB_PER_DPS, v[:, 6] / GYRO_LSB_PER_DPS,
            v[:, 3] / 340.0 + 36.53,
        ])

        self.csv.write("".join(
            f"{t[i]:.4f},{rx_time:.3f}," + ",".join(f"{d[c, i]:.5f}" for c in range(7)) + "\n"
            for i in range(n)))

        with self.lock:
            self.t = np.concatenate([self.t, t])[-MAX_SAMPLES:]
            self.data = np.concatenate([self.data, d], axis=1)[:, -MAX_SAMPLES:]
            self.packets += 1
            self.samples += n

    def snapshot(self, window_s):
        with self.lock:
            if len(self.t) == 0:
                return self.t, self.data
            i = np.searchsorted(self.t, self.t[-1] - window_s)
            return self.t[i:], self.data[:, i:]


class Window(QtWidgets.QWidget):
    def __init__(self, rx):
        super().__init__()
        self.rx = rx
        self.setWindowTitle("MPU6050 live")
        layout = QtWidgets.QVBoxLayout(self)

        conn = QtWidgets.QHBoxLayout()
        conn.addWidget(QtWidgets.QLabel("Pico IP:"))
        self.ip_box = QtWidgets.QLineEdit()
        self.ip_box.setPlaceholderText("e.g. 192.168.1.42 (printed by the Pico)")
        self.ip_box.returnPressed.connect(self.connect_pico)
        conn.addWidget(self.ip_box, 1)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_pico)
        conn.addWidget(self.connect_btn)
        self.conn_label = QtWidgets.QLabel("Not connected")
        conn.addWidget(self.conn_label)
        layout.addLayout(conn)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Window:"))
        self.window_box = QtWidgets.QComboBox()
        for s in (5, 10, 30, 60, 120):
            self.window_box.addItem(f"{s} s", s)
        self.window_box.setCurrentIndex(1)
        top.addWidget(self.window_box)
        self.status = QtWidgets.QLabel()
        top.addWidget(self.status, 1)
        layout.addLayout(top)

        self.plots = []
        for default in (0, 3):
            box = QtWidgets.QComboBox()
            box.addItems(CHANNELS)
            box.setCurrentIndex(default)
            plot = pg.PlotWidget()
            plot.showGrid(x=True, y=True)
            plot.setLabel("bottom", "time (s)")
            curve = plot.plot(pen=pg.mkPen(width=2))
            box.currentIndexChanged.connect(lambda _, p=plot, b=box: self.set_label(p, b))
            self.set_label(plot, box)
            layout.addWidget(box)
            layout.addWidget(plot, 1)
            self.plots.append((box, plot, curve))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)
        self.last_count = (time.time(), 0)
        self.rate = 0.0

    def connect_pico(self):
        ip = self.ip_box.text().strip()
        try:
            socket.inet_aton(ip)
            if ip.count(".") != 3:
                raise OSError
        except OSError:
            self.conn_label.setText("Invalid IP address")
            return
        self.rx.target = (ip, PORT)
        self.conn_label.setText(f"Connecting to {ip}...")

    @staticmethod
    def set_label(plot, box):
        plot.setLabel("left", box.currentText())

    def refresh(self):
        t, data = self.rx.snapshot(self.window_box.currentData())
        if len(t):
            for box, _, curve in self.plots:
                curve.setData(t, data[box.currentIndex()])
        now = time.time()
        dt = now - self.last_count[0]
        if dt >= 1.0:
            self.rate = (self.rx.samples - self.last_count[1]) / dt
            self.last_count = (now, self.rx.samples)
        if self.rx.target and self.rate > 0:
            self.conn_label.setText(f"Connected to {self.rx.target[0]}")
        self.status.setText(
            f"{self.rate:.0f} samples/s | packets {self.rx.packets} | lost {self.rx.lost} | "
            f"saving to {self.rx.filename}")


def main():
    rx = Receiver()
    rx.start()
    app = QtWidgets.QApplication(sys.argv)
    win = Window(rx)
    win.resize(1000, 700)
    win.show()
    code = app.exec()
    rx.stop = True
    sys.exit(code)


if __name__ == "__main__":
    main()
