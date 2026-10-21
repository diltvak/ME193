"""LEGO Education BLE gateway.

Runs in its own container (it needs the host's D-Bus socket for BlueZ). Connects to
the four LEGO devices identified by one Connection Card, then serves a single TCP
client (the App's main.py) on port 5555 using newline-delimited JSON:

  gateway -> client, every PUBLISH_PERIOD_S:
    {"type": "state", "controller": {...}, "color": {...}, "imu": {...}, "pusher": {...}}
  client -> gateway:
    {"cmd": "pusher_run_for_time", "ms": 500, "direction": "clockwise", "speed": 100}
    {"cmd": "pusher_stop"}
"""

import json
import logging
import math
import os
import socket
import threading
import time

import legoeducation as le
from legoeducation.basic_device import VersionMismatchError

logging.basicConfig(level=logging.INFO, format="[lego-ble] %(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lego-ble")

PORT = 5555
PUBLISH_PERIOD_S = 0.02
NOTIFICATION_DELAY_MS = 20  # SDK default is 100 ms; 15 is the minimum
RECONNECT_PAUSE_S = 2.0

CARD_COLORS = {
    "red": le.LEGO_COLOR_RED,
    "yellow": le.LEGO_COLOR_YELLOW,
    "blue": le.LEGO_COLOR_BLUE,
    "green": le.LEGO_COLOR_GREEN,
    "purple": le.LEGO_COLOR_PURPLE,
    "magenta": le.LEGO_COLOR_MAGENTA,
    "azure": le.LEGO_COLOR_AZURE,
    "orange": le.LEGO_COLOR_ORANGE,
}
DIRECTIONS = {
    "clockwise": le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
    "counterclockwise": le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE,
}

CARD_COLOR = CARD_COLORS[os.environ.get("LEGO_CARD_COLOR", "purple").strip().lower()]
CARD_SERIAL = os.environ.get("LEGO_CARD_SERIAL", "6235").strip()

controller = le.Controller()
colorsensor = le.ColorSensor()
doublemotor = le.DoubleMotor()  # held in the hand: only its IMU is used
singlemotor = le.SingleMotor()  # drives the pushing mechanism

ALL_DEVICES = {
    "controller": controller,
    "colorsensor": colorsensor,
    "doublemotor": doublemotor,
    "singlemotor": singlemotor,
}
# LEGO_DEVICES=controller,singlemotor limits which devices are searched for.
_wanted = [n.strip() for n in os.environ.get("LEGO_DEVICES", ",".join(ALL_DEVICES)).split(",") if n.strip()]
DEVICES = {n: ALL_DEVICES[n] for n in _wanted}

def _num(value):
    """SDK fields start as NaN until the first notification; JSON wants None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


class ImuPeakTracker:
    """Tracks the largest acceleration change since the last publish.

    The baseline is a slow moving average of |acceleration| (gravity while still), so
    the peak deviation is orientation-independent and spikes on a fast punch.
    """

    BASELINE_ALPHA = 0.05

    def __init__(self):
        self._lock = threading.Lock()
        self._baseline = None
        self._peak_acc = 0.0
        self._peak_gyro = 0.0

    def on_notification(self, _notification):
        imu = doublemotor.imu_device
        ax, ay, az = _num(imu.accelerometerX), _num(imu.accelerometerY), _num(imu.accelerometerZ)
        gx, gy, gz = _num(imu.gyroscopeX), _num(imu.gyroscopeY), _num(imu.gyroscopeZ)
        if None in (ax, ay, az, gx, gy, gz):
            return
        acc = math.sqrt(ax * ax + ay * ay + az * az)
        gyro = math.sqrt(gx * gx + gy * gy + gz * gz)
        with self._lock:
            if self._baseline is None:
                self._baseline = acc
            deviation = abs(acc - self._baseline)
            self._baseline += self.BASELINE_ALPHA * (acc - self._baseline)
            self._peak_acc = max(self._peak_acc, deviation)
            self._peak_gyro = max(self._peak_gyro, gyro)

    def take_peaks(self):
        with self._lock:
            peaks = (self._peak_acc, self._peak_gyro)
            self._peak_acc = 0.0
            self._peak_gyro = 0.0
        return peaks


imu_tracker = ImuPeakTracker()
doublemotor.set_notification_callback(imu_tracker.on_notification)


_last_lever_log = 0.0


def on_controller_notification(_notification):
    """Print raw lever values as they arrive (max 5 per second)."""
    global _last_lever_log
    now = time.monotonic()
    if now - _last_lever_log < 0.2:
        return
    _last_lever_log = now
    c = controller.sensor
    log.info("JOYSTICK raw: left=%s right=%s", _num(c.leftPercent), _num(c.rightPercent))


controller.set_notification_callback(on_controller_notification)


def connect_loop():
    """Keep every device connected; retry the ones that drop."""
    while True:
        for name, dev in DEVICES.items():
            if dev.connected:
                continue
            log.info("Searching for %s (card color=%s serial=%s)...", name, CARD_COLOR, CARD_SERIAL)
            try:
                dev.connect(
                    card_color=CARD_COLOR,
                    card_serial=CARD_SERIAL,
                    device_notification_delay=NOTIFICATION_DELAY_MS,
                )
            except VersionMismatchError as e:
                log.error("FIRMWARE MISMATCH on %s: %s", name, e)
                log.error("-> Update the %s firmware at https://code.legoeducation.com/", name)
            except Exception:
                log.exception("Error connecting to %s", name)
            log.info("%s connected=%s", name, dev.connected)
        time.sleep(RECONNECT_PAUSE_S)


def build_state():
    c, s, imu = controller.sensor, colorsensor.sensor, doublemotor.imu_device
    peak_acc, peak_gyro = imu_tracker.take_peaks()
    return {
        "type": "state",
        "t": time.time(),
        "controller": {
            "connected": controller.connected,
            "left": _num(c.leftPercent),
            "right": _num(c.rightPercent),
        },
        "color": {
            "connected": colorsensor.connected,
            "color": _num(s.color),
            "hue": _num(s.hue),
            "reflection": _num(s.reflection),
        },
        "imu": {
            "connected": doublemotor.connected,
            "acc": [_num(imu.accelerometerX), _num(imu.accelerometerY), _num(imu.accelerometerZ)],
            "gyro": [_num(imu.gyroscopeX), _num(imu.gyroscopeY), _num(imu.gyroscopeZ)],
            "peak_acc": peak_acc,
            "peak_gyro": peak_gyro,
        },
        "pusher": {"connected": singlemotor.connected},
    }


def pusher_stop():
    if singlemotor.connected:
        singlemotor.motor_stop(blocking=False)


def handle_command(msg):
    cmd = msg.get("cmd")
    if cmd == "pusher_run_for_time":
        if not singlemotor.connected:
            log.warning("Pusher command ignored: single motor not connected")
            return
        singlemotor.motor_run_for_time(
            int(msg["ms"]),
            direction=DIRECTIONS[msg["direction"]],
            speed=int(msg.get("speed", 100)),
            blocking=False,
        )
    elif cmd == "pusher_stop":
        pusher_stop()
    else:
        log.warning("Unknown command: %r", msg)


def serve_client(conn):
    stop = threading.Event()

    def publisher():
        while not stop.is_set():
            try:
                conn.sendall((json.dumps(build_state()) + "\n").encode())
            except OSError:
                stop.set()
                return
            time.sleep(PUBLISH_PERIOD_S)

    threading.Thread(target=publisher, daemon=True).start()
    buf = b""
    try:
        while not stop.is_set():
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    try:
                        handle_command(json.loads(line))
                    except Exception:
                        log.exception("Failed to handle command %r", line)
    except OSError:
        pass
    finally:
        stop.set()
        conn.close()
        try:
            pusher_stop()  # never leave the pusher running without a controller
        except Exception:
            log.exception("Failed to stop pusher after client disconnect")


def main():
    threading.Thread(target=connect_loop, daemon=True).start()
    server = socket.create_server(("0.0.0.0", PORT), reuse_port=True)
    log.info("Gateway listening on :%d", PORT)
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.info("App connected from %s", addr)
        serve_client(conn)  # one client at a time: the App's main.py
        log.info("App disconnected")


if __name__ == "__main__":
    main()
