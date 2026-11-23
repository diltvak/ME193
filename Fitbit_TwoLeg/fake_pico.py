"""Pretend to be one or two Picos on this computer, for testing receiver.py without hardware.

Replays a recorded CSV (pc_receiver.py format) with the exact UDP protocol of pico_main.py:
waits for "hello", streams <seq><n> + n x <ticks_us><14 raw MPU bytes> at 100 Hz, stops after
5 s without a hello. The file loops forever.

    python fake_pico.py                                   # left on :5006, right on :5007
    python receiver.py --left 127.0.0.1:5006 --right 127.0.0.1:5007

    python fake_pico.py --loss 0.05                       # drop 5% of packets
    python fake_pico.py --file ../Fitbit/trial_3.csv

The right leg replays the same walk shifted by half a stride (0.55 s), with gy/gz mirrored,
so the two traces alternate like real legs. Each fake Pico starts its clock at a random value
so the receiver's clock alignment and 30-bit wraparound are exercised too.
"""
from __future__ import annotations

import argparse
import random
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_FILE = HERE.parent / "Fitbit" / "trial_1.csv"
ACCEL_LSB_PER_G = 8192.0
GYRO_LSB_PER_DPS = 65.5
TICKS_MOD = 1 << 30
BATCH = 5
HELLO_TIMEOUT_S = 5.0


def load_raw(path, shift_s=0.0, mirror=False):
    """CSV -> (dt_us per sample, (n, 7) int16 rows in MPU order ax ay az temp gx gy gz)."""
    d = np.genfromtxt(path, delimiter=",", skip_header=1, invalid_raise=False)
    d = d[np.isfinite(d).all(axis=1)]
    t = d[:, 0]
    acc, gyr, temp = d[:, 2:5], d[:, 5:8].copy(), d[:, 8]
    if mirror:
        gyr[:, 1:] *= -1
    raw = np.column_stack([acc * ACCEL_LSB_PER_G, (temp - 36.53) * 340.0, gyr * GYRO_LSB_PER_DPS])
    raw = np.clip(np.round(raw), -32768, 32767).astype(">i2")
    dt = np.diff(t, append=t[-1] + 0.01)
    dt = np.where((dt > 0) & (dt < 0.5), dt, 0.01)            # file gaps -> nominal step on loop
    k = int(np.searchsorted(np.cumsum(dt), shift_s))
    return np.roll(dt, -k), np.roll(raw, -k, axis=0)


def fake_pico(name, port, path, shift_s, mirror, loss, stop):
    dt, raw = load_raw(path, shift_s, mirror)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.005)
    print(f"[{name}] fake Pico on 127.0.0.1:{port}, replaying {Path(path).name} ({len(raw)} samples)")
    ticks = random.randrange(TICKS_MOD)
    addr, last_hello, seq, i = None, 0.0, 0, 0
    next_t = time.perf_counter()
    batch = []
    while not stop.is_set():
        try:
            _, sender = sock.recvfrom(16)
            if addr is None:
                print(f"[{name}] PC connected: {sender[0]}:{sender[1]}")
                batch = []
                next_t = time.perf_counter()
            addr, last_hello = sender, time.time()
        except (socket.timeout, OSError):
            pass
        if addr and time.time() - last_hello > HELLO_TIMEOUT_S:
            print(f"[{name}] PC timed out")
            addr = None
        if addr is None:
            continue
        while time.perf_counter() >= next_t:                     # "timer callback": take samples
            batch.append(struct.pack("<I", ticks) + raw[i].tobytes())
            step = dt[i]
            ticks = (ticks + int(step * 1e6)) % TICKS_MOD
            next_t += step
            i = (i + 1) % len(raw)
        if len(batch) >= BATCH:
            pkt = struct.pack("<IB", seq, len(batch)) + b"".join(batch)
            if random.random() >= loss:
                sock.sendto(pkt, addr)
            seq = (seq + 1) & 0xFFFFFFFF
            batch = []
    sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE, help="recorded CSV to replay")
    ap.add_argument("--left-port", type=int, default=5006)
    ap.add_argument("--right-port", type=int, default=5007)
    ap.add_argument("--only", choices=["left", "right"], help="run just one fake Pico")
    ap.add_argument("--loss", type=float, default=0.0, help="fraction of packets to drop")
    args = ap.parse_args()

    stop = threading.Event()
    legs = [("left", args.left_port, 0.0, False), ("right", args.right_port, 0.55, True)]
    threads = [threading.Thread(target=fake_pico, args=(n, p, args.file, s, m, args.loss, stop), daemon=True)
               for n, p, s, m in legs if args.only in (None, n)]
    for th in threads:
        th.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
