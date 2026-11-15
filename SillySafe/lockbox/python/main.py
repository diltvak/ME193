"""Finger Lockbox - Linux side.

Press the button, then show the secret sequence of finger counts to the camera
one at a time, holding each for about a second. If the sequence matches CODE,
the servo opens the box. Press the button again to lock it.

The LED matrix shows: "?" while watching, each digit as it is accepted, a
check mark when the box opens and an X when the sequence is wrong or times out.
"""
import glob
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from arduino.app_utils import App, Bridge

# --- Settings --------------------------------------------------------------
CODE = [0, 2, 1, 5]   # the secret sequence. Neighbouring entries must differ.
HOLD_S = 1.0          # hold a finger count steady this long for it to register
MIN_CONF = 0.8        # ignore predictions the model is less sure of than this
TIMEOUT_S = 30.0      # give up if the sequence isn't finished in this time
SMOOTH = 3            # average the model's output over this many frames
RESULT_S = 2.0        # how long the X stays on the matrix

assert all(0 <= n <= 5 for n in CODE), "CODE entries must be 0-5"
assert all(a != b for a, b in zip(CODE, CODE[1:])), "neighbouring CODE entries must differ"

BLANK, QUESTION, CHECK, CROSS = 10, 11, 12, 13   # codes for the sketch's show()
SIZE = 128

opts = ort.SessionOptions()
opts.intra_op_num_threads = os.cpu_count() or 4
session = ort.InferenceSession(str(Path(__file__).with_name("fingers.onnx")), opts,
                               providers=["CPUExecutionProvider"])

cap = None
state = "starting"    # starting -> locked -> listening -> open -> locked ...
entries = []          # finger counts accepted so far
history = deque(maxlen=SMOOTH)
candidate, candidate_since = None, 0.0
listen_start = 0.0
blank_at = None       # when to clear the X from the matrix


def log(msg):
    print(msg, flush=True)


def open_camera():
    """Open the first video device that actually delivers a frame."""
    for dev in sorted(glob.glob("/dev/video*")):
        c = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if c.read()[0]:
                log(f"Camera opened: {dev}")
                return c
        c.release()
    return None


def predict(frame):
    """Return (finger count, confidence). Same preprocessing as the training data."""
    h, w = frame.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    img = cv2.resize(frame[y:y + s, x:x + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    logits = session.run(None, {"image": img.transpose(2, 0, 1)[None]})[0][0]
    p = np.exp(logits - logits.max())
    history.append(p / p.sum())
    probs = np.mean(history, axis=0)
    return int(probs.argmax()), float(probs.max())


def lock(icon=BLANK):
    global state, blank_at
    Bridge.call("set_servo", False)
    Bridge.call("show", icon)
    blank_at = time.time() + RESULT_S if icon != BLANK else None
    state = "locked"


def start_listening():
    global state, entries, candidate, listen_start, blank_at
    entries, candidate, blank_at = [], None, None
    history.clear()
    listen_start = time.time()
    Bridge.call("show", QUESTION)
    state = "listening"
    log("Button pressed: watching for the sequence")


def watch(frame):
    """Register finger counts held steady; open or reject once enough are in."""
    global state, candidate, candidate_since
    now = time.time()
    if now - listen_start > TIMEOUT_S:
        log("Timed out")
        lock(CROSS)
        return

    n, conf = predict(frame)
    if conf < MIN_CONF:
        candidate = None
        return
    if n != candidate:
        candidate, candidate_since = n, now
        return
    if now - candidate_since < HOLD_S or (entries and entries[-1] == n):
        return

    entries.append(n)
    Bridge.call("show", n)
    log(f"Accepted {n}  ({len(entries)}/{len(CODE)})")
    if len(entries) < len(CODE):
        return
    if entries == CODE:
        Bridge.call("set_servo", True)
        Bridge.call("show", CHECK)
        state = "open"
        log("Correct sequence: box open. Press the button to lock it.")
    else:
        log("Wrong sequence")
        lock(CROSS)


def step():
    global cap, blank_at
    if state == "starting":
        lock()   # make sure the servo starts in the locked position
        log("Locked. Press the button to enter the sequence.")

    if cap is None:
        cap = open_camera()
        if cap is None:
            log("No camera found. Plug in the USB camera, then restart the app.")
            time.sleep(2)
            return

    ok, frame = cap.read()   # read every pass so frames are never stale
    if not ok:
        log("Camera read failed, reopening")
        cap.release()
        cap = None
        return

    if blank_at and time.time() > blank_at:
        Bridge.call("show", BLANK)
        blank_at = None

    if Bridge.call("take_button"):
        if state == "open":
            lock()
            log("Locked")
        else:
            start_listening()   # also restarts an attempt already in progress
    elif state == "listening":
        watch(frame)


def loop():
    # The MCU may still be starting right after a flash, so a failed Bridge
    # call is retried on the next pass instead of crashing the app.
    try:
        step()
    except Exception as e:
        log(f"Retrying after error: {e!r}")
        time.sleep(0.5)


App.run(user_loop=loop)
