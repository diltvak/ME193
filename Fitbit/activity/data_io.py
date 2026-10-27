"""Data input: CSV loading, resampling/windowing, and live sample sources.

Every source (UdpSource, SimulatedSource, ReplaySource) exposes the same interface:

    start()            begin producing samples
    read() -> (t, X)   drain everything received since the last call
                       t: (n,) seconds, X: (n, 7) = ax ay az (g), gx gy gz (dps), temp (C)
    status() -> dict   connected, packets, lost, bad, samples
    truth              ground-truth label if known (sim/replay), else None
    stop()
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .config import Config, FILE_LABELS

CSV_HEADER = "t_s,pc_time,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_c"
N_CH = 7        # ax ay az gx gy gz temp; features only use the first 6


# ----------------------------------------------------------------------------
# CSV + resampling
# ----------------------------------------------------------------------------
def load_csv(path):
    """Load a recorded CSV -> (t, X) with X columns ax ay az gx gy gz temp. Bad rows are dropped."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, invalid_raise=False)
    raw = np.atleast_2d(raw)
    raw = raw[np.isfinite(raw).all(axis=1)]
    return raw[:, 0], raw[:, 2:9]


def save_csv(path, t, X, pc_time=None):
    """Write samples in the same format pc_receiver.py produces."""
    if pc_time is None:
        pc_time = np.full(len(t), time.time())
    with open(path, "w") as f:
        f.write(CSV_HEADER + "\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.4f},{pc_time[i]:.3f}," + ",".join(f"{v:.5f}" for v in X[i]) + "\n")


def discover_files(data_dir):
    """Find '<label>_data_<n>.csv' files -> {class label: [paths]}."""
    out = {}
    for p in sorted(Path(data_dir).glob("*_data_*.csv")):
        prefix = p.name.split("_data_")[0].lower()
        if prefix in FILE_LABELS:
            out.setdefault(FILE_LABELS[prefix], []).append(p)
    return out


def resample_uniform(t, X, cfg: Config):
    """Interpolate irregular samples onto a fixed-rate grid.

    Handles jitter and dropped packets. Gaps up to cfg.max_gap_s are bridged by linear
    interpolation; grid points inside longer gaps are flagged invalid so callers can
    refuse to build windows across them.

    Returns (t_u, X_u, valid); valid[i] is False if grid point i lies in a gap > max_gap_s.
    """
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    if len(t) < 2:
        return t, X, np.ones(len(t), bool)
    keep = np.concatenate([[True], np.diff(np.maximum.accumulate(t)) > 0])   # strictly increasing
    t, X = t[keep], X[keep]
    grid = np.arange(t[0], t[-1], 1.0 / cfg.fs)
    X_u = np.column_stack([np.interp(grid, t, X[:, c]) for c in range(X.shape[1])])
    hi = np.clip(np.searchsorted(t, grid, side="right"), 1, len(t) - 1)
    valid = (t[hi] - t[hi - 1]) <= cfg.max_gap_s
    return grid, X_u, valid


def sliding_windows(X_u, valid, cfg: Config):
    """Yield (start_index, window) for every fully-valid window (n = window_n, hop = hop_n)."""
    n, hop = cfg.window_n, cfg.hop_n
    for s in range(0, len(X_u) - n + 1, hop):
        if valid[s:s + n].all():
            yield s, X_u[s:s + n]


# ----------------------------------------------------------------------------
# Live source: Pico over Wi-Fi/UDP
# ----------------------------------------------------------------------------
class UdpSource:
    """Receives the Pico's UDP batches on a background thread.

    Mirrors pc_receiver.py's protocol: a "hello" packet (repeated every second) tells the
    Pico where to stream; each packet is <seq u32><n u8> + n x (<ticks_us u32> + 14 raw
    big-endian MPU bytes).

    Buffering: the Pico buffers in a ring on its side (pico_main.py). On the PC the socket
    is read by a dedicated thread with a large OS receive buffer and pushed into a bounded
    deque, so slow feature extraction or plotting in the main thread never stalls reception
    or overflows the socket. Because hellos are sent continuously, a Pico that timed out or
    a PC that restarted reconnects by itself.
    """
    name = "udp"
    truth = None
    _REC = np.dtype([("t", "<u4"), ("v", ">i2", (7,))])   # raw order: ax ay az temp gx gy gz
    _HDR = struct.Struct("<IB")
    _TICKS_MOD = 1 << 30                                   # MicroPython ticks_us wraps at 2**30

    def __init__(self, cfg: Config, ip=None):
        self.cfg = cfg
        self.target = (ip or cfg.pico_ip, cfg.pico_port)
        self._chunks = deque(maxlen=4000)                  # ~200 s of packets if never drained
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.packets = self.lost = self.bad = self.samples = 0
        self._last_seq = None
        self._last_raw = None
        self._abs_us = 0.0
        self._last_rx = 0.0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def read(self):
        with self._lock:
            chunks = list(self._chunks)
            self._chunks.clear()
        if not chunks:
            return np.empty(0), np.empty((0, N_CH))
        return np.concatenate([c[0] for c in chunks]), np.vstack([c[1] for c in chunks])

    def status(self):
        return {"connected": (time.time() - self._last_rx) < self.cfg.stale_s,
                "packets": self.packets, "lost": self.lost, "bad": self.bad,
                "samples": self.samples}

    def _run(self):
        sock = None
        last_hello = 0.0
        while not self._stop.is_set():
            if sock is None:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
                    except OSError:
                        pass
                    sock.bind(("0.0.0.0", self.cfg.local_port))
                    sock.settimeout(0.2)
                except OSError as e:                      # port busy (is pc_receiver.py still open?)
                    print(f"[udp] cannot bind port {self.cfg.local_port}: {e}; retrying")
                    sock = None
                    time.sleep(1.0)
                    continue
            if time.time() - last_hello >= self.cfg.hello_period_s:
                try:
                    sock.sendto(b"hello", self.target)
                except OSError:
                    pass                                   # network down; keep trying
                last_hello = time.time()
            try:
                pkt, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:                                # e.g. network interface went away
                sock.close()
                sock = None
                time.sleep(0.5)
                continue
            try:
                self._handle(pkt)
            except Exception:                              # malformed packet: count and move on
                self.bad += 1
        if sock:
            sock.close()

    def _handle(self, pkt):
        seq, n = self._HDR.unpack_from(pkt)
        if n == 0 or len(pkt) != self._HDR.size + n * self._REC.itemsize:
            self.bad += 1
            return
        rec = np.frombuffer(pkt, dtype=self._REC, count=n, offset=self._HDR.size)

        reboot = self._last_seq is not None and (seq == 0 or self._last_seq - seq > 1000)
        if reboot:                                         # Pico restarted: new time base
            self._last_raw = None
            self._last_seq = None
            self._abs_us += 1e6                            # force a segment break
        if self._last_seq is not None:
            if seq <= self._last_seq:                      # duplicate / reordered
                return
            self.lost += seq - self._last_seq - 1
        self._last_seq = seq

        raw = rec["t"].astype(np.int64)
        prev = raw[0] if self._last_raw is None else self._last_raw
        steps = np.diff(raw, prepend=prev) % self._TICKS_MOD   # unwrap 30-bit microsecond counter
        times_us = self._abs_us + np.cumsum(steps)
        self._abs_us = times_us[-1]
        self._last_raw = raw[-1]
        t = times_us / 1e6

        v = rec["v"].astype(np.float64)
        c = self.cfg
        X = np.column_stack([v[:, 0] / c.accel_lsb_per_g, v[:, 1] / c.accel_lsb_per_g,
                             v[:, 2] / c.accel_lsb_per_g,
                             v[:, 4] / c.gyro_lsb_per_dps, v[:, 5] / c.gyro_lsb_per_dps,
                             v[:, 6] / c.gyro_lsb_per_dps, v[:, 3] / 340.0 + 36.53])
        with self._lock:
            self._chunks.append((t, X))
            self.packets += 1
            self.samples += n
            self._last_rx = time.time()


# ----------------------------------------------------------------------------
# Simulated source (no hardware needed)
# ----------------------------------------------------------------------------
class SimulatedSource:
    """Synthetic ankle IMU that follows a schedule of activities in real time.

    Mimics the stream's flaws: timestamp jitter and randomly dropped 5-sample packets,
    with a random sensor orientation. `truth` is the activity currently being simulated.
    The waveforms are crude stand-ins (smooth sinusoid for biking, harmonic-rich swing plus
    impact pulses for walking/running); they test the plumbing, not real-world accuracy.
    """
    name = "sim"
    # label: (stride Hz, gyro amplitude dps, impact g, impact width s, sway g, harmonic mix)
    PARAMS = {
        "standing": (0.0, 0.0, 0.0, 0.03, 0.0, ()),
        "walking": (0.9, 150.0, 2.6, 0.04, 0.65, (0.8, 0.6, 0.4)),
        "running": (1.4, 220.0, 5.5, 0.03, 0.95, (0.8, 0.6, 0.4)),
        "biking": (1.1, 200.0, 0.0, 0.03, 0.55, ()),
    }

    def __init__(self, cfg: Config, schedule=None, seed=0, drop_prob=0.03, speed=1.0, loop=True):
        self.cfg = cfg
        self.schedule = schedule or [("standing", 20), ("walking", 20), ("running", 20), ("biking", 20)]
        self.seed, self.drop_prob, self.speed, self.loop = seed, drop_prob, speed, loop
        rng = np.random.default_rng(seed)
        self._gravity = self._unit(rng.normal(size=3))
        self._axis = self._unit(rng.normal(size=3))          # dominant rotation axis
        self._impact_dir = self._unit(rng.normal(size=3))
        self._noise = np.random.default_rng(seed + 1)
        self._bounds = np.cumsum([d for _, d in self.schedule])
        self._k = 0
        self._t0 = None
        self.truth = self.schedule[0][0]
        self.samples = 0

    @staticmethod
    def _unit(v):
        return v / np.linalg.norm(v)

    def start(self):
        self._t0 = time.time()

    def stop(self):
        pass

    def status(self):
        return {"connected": True, "packets": self.samples // 5, "lost": 0, "bad": 0,
                "samples": self.samples}

    def _bout(self, t):
        total = self._bounds[-1]
        if not self.loop and t >= total:
            return len(self.schedule) - 1
        loops, tm = divmod(t, total)
        return int(np.searchsorted(self._bounds, tm, side="right")) + int(loops) * len(self.schedule)

    def read(self):
        k_end = int((time.time() - self._t0) * self.speed * self.cfg.fs)
        if k_end <= self._k:
            return np.empty(0), np.empty((0, N_CH))
        k = np.arange(self._k, k_end)
        self._k = k_end
        t = k / self.cfg.fs + self._noise.uniform(-0.001, 0.001, len(k))
        t = np.maximum.accumulate(t)
        keep = np.ones(len(k), bool)                          # drop whole 5-sample "packets"
        for s in range(0, len(k), 5):
            if self._noise.random() < self.drop_prob:
                keep[s:s + 5] = False
        t = t[keep]
        if len(t) == 0:
            return np.empty(0), np.empty((0, N_CH))
        bouts = np.array([self._bout(x) for x in t[[0, -1]]])
        X = np.empty((len(t), N_CH))
        # a chunk is ~50 ms, so it rarely spans two bouts; per-sample lookup keeps it exact anyway
        per_sample = np.array([self._bout(x) for x in t]) if bouts[0] != bouts[1] else np.full(len(t), bouts[0])
        for b in np.unique(per_sample):
            m = per_sample == b
            X[m] = self._synth(self.schedule[b % len(self.schedule)][0], t[m], b)
        self.truth = self.schedule[per_sample[-1] % len(self.schedule)][0]
        self.samples += len(t)
        return t, X

    def _synth(self, label, t, bout):
        f0, gyro_amp, imp_amp, imp_w, sway, harm = self.PARAMS[label]
        rng = np.random.default_rng(self.seed + 100 + bout)
        f = f0 * (1 + 0.1 * rng.uniform(-1, 1))
        n = len(t)
        acc = np.tile(self._gravity, (n, 1)) + self._noise.normal(0, 0.004, (n, 3))
        gyro = self._noise.normal(0, 1.7, (n, 3)) + np.array([-2.0, 2.5, -1.8])
        if f0 > 0:
            ph = 2 * np.pi * f * t
            wave = np.sin(ph) + sum(a * np.sin((i + 2) * ph + 0.7) for i, a in enumerate(harm))
            gyro += gyro_amp * wave[:, None] * self._axis
            acc += sway * np.sin(ph + 1.0)[:, None] * self._gravity
            if imp_amp:
                dt = ((t * f + 0.5) % 1.0 - 0.5) / f          # time from nearest "heel strike"
                acc += imp_amp * np.exp(-0.5 * (dt / imp_w) ** 2)[:, None] * self._impact_dir
            acc += self._noise.normal(0, 0.03, (n, 3))
            gyro += self._noise.normal(0, 4.0, (n, 3))
        return np.column_stack([acc, gyro, np.full(n, 26.0)])


# ----------------------------------------------------------------------------
# Replay recorded CSVs as if they were live
# ----------------------------------------------------------------------------
class ReplaySource:
    """Plays labeled CSV files back in real time (gaps included), for testing the live loop."""
    name = "replay"

    def __init__(self, cfg: Config, files, speed=1.0):
        """files: list of (label, path)."""
        self.cfg, self.speed = cfg, speed
        ts, xs, ls, offset = [], [], [], 0.0
        for label, path in files:
            t, X = load_csv(path)
            t = t - t[0] + offset
            offset = t[-1] + 0.5
            ts.append(t), xs.append(X), ls.append(np.full(len(t), label, dtype=object))
        self._t, self._X, self._labels = np.concatenate(ts), np.vstack(xs), np.concatenate(ls)
        self._i = 0
        self._t0 = None
        self.truth = self._labels[0]
        self.samples = 0

    def start(self):
        self._t0 = time.time()

    def stop(self):
        pass

    def status(self):
        done = self._i >= len(self._t)
        return {"connected": not done, "packets": self.samples // 5, "lost": 0, "bad": 0,
                "samples": self.samples}

    def read(self):
        now = (time.time() - self._t0) * self.speed
        j = int(np.searchsorted(self._t, now, side="right"))
        if j <= self._i:
            return np.empty(0), np.empty((0, N_CH))
        t, X = self._t[self._i:j], self._X[self._i:j]
        self.truth = self._labels[j - 1]
        self._i = j
        self.samples += len(t)
        return t, X


def make_source(kind, cfg: Config, ip=None, replay_files=None, schedule=None, speed=1.0):
    """Factory used by collect.py and live.py."""
    if kind == "udp":
        return UdpSource(cfg, ip)
    if kind == "sim":
        return SimulatedSource(cfg, schedule=schedule, speed=speed)
    if kind == "replay":
        return ReplaySource(cfg, replay_files, speed=speed)
    raise ValueError(f"unknown source {kind!r}")
