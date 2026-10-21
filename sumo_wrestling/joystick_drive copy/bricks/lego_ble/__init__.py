"""Client for the LEGO BLE gateway service (see gateway.py).

Usage from the App's main.py:

    from lego_ble import LegoGateway
    lego = LegoGateway()
    state = lego.snapshot()          # latest device data (dict)
    lego.pusher_run_for_time(500, "clockwise", 100)
"""

import copy
import json
import socket
import threading
import time

from arduino.app_utils import Logger

log = Logger("lego_ble")

DEFAULT_HOST = "lego-ble"  # compose service name from brick_compose.yaml
DEFAULT_PORT = 5555
STALE_AFTER_S = 0.5  # no state for this long => treat every device as disconnected

_DISCONNECTED_STATE = {
    "controller": {"connected": False, "left": None, "right": None},
    "color": {"connected": False, "color": None, "hue": None, "reflection": None},
    "imu": {"connected": False, "acc": [None] * 3, "gyro": [None] * 3, "peak_acc": 0.0, "peak_gyro": 0.0},
    "pusher": {"connected": False},
}


class LegoGateway:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._sock = None
        self._state = copy.deepcopy(_DISCONNECTED_STATE)
        self._last_rx = 0.0
        # IMU peaks are accumulated between snapshots so no punch is lost between ticks.
        self._peak_acc = 0.0
        self._peak_gyro = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    # ---- public API -------------------------------------------------------

    @property
    def gateway_connected(self) -> bool:
        return self._sock is not None and time.monotonic() - self._last_rx < STALE_AFTER_S

    def snapshot(self) -> dict:
        """Latest device state; IMU peaks are those seen since the previous snapshot."""
        with self._lock:
            if self._sock is None or time.monotonic() - self._last_rx >= STALE_AFTER_S:
                return copy.deepcopy(_DISCONNECTED_STATE)
            state = copy.deepcopy(self._state)
            state["imu"]["peak_acc"] = self._peak_acc
            state["imu"]["peak_gyro"] = self._peak_gyro
            self._peak_acc = 0.0
            self._peak_gyro = 0.0
            return state

    def pusher_run_for_time(self, ms: int, direction: str, speed: int = 100):
        self._send({"cmd": "pusher_run_for_time", "ms": int(ms), "direction": direction, "speed": int(speed)})

    def pusher_stop(self):
        self._send({"cmd": "pusher_stop"})

    # ---- internals --------------------------------------------------------

    def _send(self, msg: dict):
        sock = self._sock
        if sock is None:
            log.warning("LEGO gateway not connected; dropped %s", msg.get("cmd"))
            return
        try:
            sock.sendall((json.dumps(msg) + "\n").encode())
        except OSError as e:
            log.warning("Send to LEGO gateway failed: %s", e)

    def _run(self):
        while True:
            try:
                sock = socket.create_connection((self._host, self._port), timeout=3)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(2)
                self._sock = sock
                log.info("Connected to LEGO gateway at %s:%d", self._host, self._port)
                self._read_loop(sock)
            except OSError as e:
                log.debug("LEGO gateway connection error: %s", e)
            finally:
                if self._sock is not None:
                    log.warning("Lost connection to LEGO gateway; retrying")
                self._sock = None
            time.sleep(1)

    def _read_loop(self, sock):
        buf = b""
        while True:
            data = sock.recv(65536)
            if not data:
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("type") != "state":
                    continue
                with self._lock:
                    self._state = msg
                    self._last_rx = time.monotonic()
                    self._peak_acc = max(self._peak_acc, msg["imu"]["peak_acc"])
                    self._peak_gyro = max(self._peak_gyro, msg["imu"]["peak_gyro"])
