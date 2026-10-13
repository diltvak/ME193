"""Minifigure detection with the YOLO model, run off the main thread.

YOLO inference is slow compared to the pose loop, so MinifigDetector runs it on a background
thread that always works on the newest frame submitted (older ones are dropped). A minifig class
"appears" once it has been seen in MINIFIG_CONFIRM_FRAMES consecutive detections, which stops
single-frame false positives from spending a once-per-game powerup. Holding the figure in view
doesn't re-trigger; it has to leave the view and come back (and wait out MINIFIG_REARM_S).
"""

import math
import threading
import time
from collections import defaultdict

from powerup_pong.config import (MINIFIG_CONF, MINIFIG_CONFIRM_FRAMES, MINIFIG_IMGSZ, MINIFIG_MAX_FPS,
                                 MINIFIG_THREADS, MINIFIG_REARM_S,
                                 NO_FIG_CLASS, WEIGHTS_PATH)


class MinifigDetector:
    def __init__(self, weights=WEIGHTS_PATH, enabled=True):
        self.status = "loading" if enabled else "off"   # off | loading | ready | error: <why>
        self.detections = []    # latest (class name, confidence, (x1, y1, x2, y2) as 0-1 fractions)
        self._weights = weights
        self._frame = None
        self._events = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._streak = defaultdict(int)
        self._last_fire = defaultdict(lambda: -math.inf)
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()

    @property
    def ready(self):
        return self.status == "ready"

    def submit(self, frame):
        """Offer the newest camera frame (unflipped BGR). It must not be modified afterward."""
        if self.status in ("loading", "ready"):
            with self._lock:
                self._frame = frame
            self._wake.set()

    def poll(self):
        """Class names that newly appeared since the last call."""
        with self._lock:
            events, self._events = self._events, []
        return events

    def close(self):
        self._stop = True
        self._wake.set()

    def _run(self):
        try:
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(MINIFIG_THREADS)
            model = YOLO(self._weights)
        except Exception as err:   # no weights or no ultralytics: play on without powerups
            self.status = f"error: {err}"
            print(f"Minifig detection disabled: {err}")
            return
        self.status = "ready"
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            started = time.monotonic()
            try:
                result = model(frame, conf=MINIFIG_CONF, imgsz=MINIFIG_IMGSZ, verbose=False)[0]
                dets = [(result.names[int(b.cls)], float(b.conf), tuple(float(v) for v in b.xyxyn[0]))
                        for b in result.boxes if result.names[int(b.cls)] != NO_FIG_CLASS]
            except Exception as err:
                self.status = f"error: {err}"
                print(f"Minifig detection disabled: {err}")
                return
            self._update(dets, time.monotonic())
            time.sleep(max(0.0, 1 / MINIFIG_MAX_FPS - (time.monotonic() - started)))   # leave CPU to the game

    def _update(self, dets, now):
        seen = {name for name, _, _ in dets}
        with self._lock:
            self.detections = dets
            for name in seen | set(self._streak):
                self._streak[name] = self._streak[name] + 1 if name in seen else 0
                if (self._streak[name] == MINIFIG_CONFIRM_FRAMES
                        and now - self._last_fire[name] >= MINIFIG_REARM_S):
                    self._last_fire[name] = now
                    self._events.append(name)
