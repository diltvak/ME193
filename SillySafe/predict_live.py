#!/usr/bin/env python3
"""
Live finger counting from a camera using model/fingers.onnx.

Runs unchanged on the Mac and on the Arduino Uno Q (Linux side). It needs only
onnxruntime, opencv and numpy -- no PyTorch.

    python3 predict_live.py                # Mac: finds "icspring camera", shows preview
    python3 predict_live.py --camera 0     # Uno Q: USB camera at /dev/video0
    python3 predict_live.py --no-window    # headless: print predictions only

Preprocessing matches capture_fingers.py exactly: centre-crop to a square,
resize to 128x128 (area interpolation), RGB, scaled to 0..1.
Predictions are smoothed over the last few frames to stop flicker.
"""
import argparse
import os
import platform
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
SIZE = 128


def open_camera(camera):
    """camera: an index ("0"), a device path ("/dev/video0") or a name (macOS)."""
    if camera.isdigit():
        return cv2.VideoCapture(int(camera))
    if camera.startswith("/dev/"):
        return cv2.VideoCapture(camera)
    if platform.system() != "Darwin":
        sys.exit("Camera names only work on macOS; pass an index or /dev/videoN.")
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True).stderr
    video = out.split("AVFoundation audio devices")[0]
    for idx, name in re.findall(r"\[(\d+)\]\s+(.*)", video):
        if camera.lower() in name.lower():
            return cv2.VideoCapture(int(idx), cv2.CAP_AVFOUNDATION)
    sys.exit(f'Camera "{camera}" not found.')


def preprocess(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    sq = frame[y:y + s, x:x + s]
    img = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return sq, img.transpose(2, 0, 1)[None]


def main():
    default_cam = "icspring camera" if platform.system() == "Darwin" else "0"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=HERE / "model" / "fingers.onnx")
    ap.add_argument("--camera", default=default_cam, help=f'index, /dev/videoN, or macOS name (default "{default_cam}")')
    ap.add_argument("--smooth", type=int, default=5, help="average probabilities over this many frames (default 5)")
    ap.add_argument("--min-conf", type=float, default=0.6, help='below this confidence report "?" (default 0.6)')
    ap.add_argument("--no-window", action="store_true", help="no preview window; print to the terminal")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = run until quit)")
    args = ap.parse_args()

    headless = args.no_window or (platform.system() == "Linux" and not os.environ.get("DISPLAY"))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    sess = ort.InferenceSession(str(args.model), opts, providers=["CPUExecutionProvider"])

    cap = open_camera(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    history = deque(maxlen=max(1, args.smooth))
    start = last_print = time.time()
    last_label, ms = None, 0.0
    print("Running. Ctrl+C to stop." if headless else "Running. Press q to quit.")
    try:
        while not (args.seconds and time.time() - start > args.seconds):
            ok, frame = cap.read()
            if not ok:
                sys.exit("Camera read failed.")
            view, x = preprocess(frame)

            t0 = time.perf_counter()
            logits = sess.run(None, {"image": x})[0][0]
            ms = 0.9 * ms + 0.1 * (time.perf_counter() - t0) * 1000 if ms else (time.perf_counter() - t0) * 1000

            p = np.exp(logits - logits.max())
            history.append(p / p.sum())
            probs = np.mean(history, axis=0)
            n, conf = int(probs.argmax()), float(probs.max())
            label = str(n) if conf >= args.min_conf else "?"

            if headless:
                now = time.time()
                if label != last_label or now - last_print > 1.0:
                    print(f"fingers: {label}   conf {conf:.2f}   {ms:.0f} ms/frame", flush=True)
                    last_label, last_print = label, now
            else:
                view = view.copy()
                cv2.putText(view, f"{label}  ({conf:.0%})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(view, f"{ms:.0f} ms", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("fingers (q to quit)", view)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
