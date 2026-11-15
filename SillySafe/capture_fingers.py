#!/usr/bin/env python3
"""
Capture training images of a hand showing N fingers from an external camera.

Usage:
    python3 capture_fingers.py                   # prompts for the finger count
    python3 capture_fingers.py 3                 # skip the prompt, label "3"
    python3 capture_fingers.py 0 --duration 60   # 60 seconds of "0 fingers"
    python3 capture_fingers.py --list            # list cameras

Images are saved to  dataset/<label>/  as small square JPEGs (default 128x128).

Controls (in the preview window):
    SPACE  pause / resume capturing
    q/ESC  quit

Post-processing applied to each saved frame:
    - center-crop to a square (cameras are 16:9; networks want square input)
    - resize to --size (default 128) with area interpolation
    - blurry frames are rejected (Laplacian variance below --min-sharpness)
    - near-duplicate frames (hand not moving) are skipped (--min-diff)
    - optional --gray to save grayscale
Tip: vary hand position, distance, angle, rotation, lighting and background
between/during runs -- that variety matters more than raw image count.
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CAMERA_NAME = "icspring camera"
HERE = Path(__file__).resolve().parent


def list_cameras():
    """Return [(index, name)] of AVFoundation video devices (same order OpenCV uses)."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    ).stderr
    cams, in_video = [], False
    for line in out.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
        elif "AVFoundation audio devices" in line:
            in_video = False
        elif in_video:
            m = re.search(r"\[(\d+)\]\s+(.*)$", line)
            if m:
                cams.append((int(m.group(1)), m.group(2).strip()))
    return cams


def find_camera(name):
    cams = list_cameras()
    for idx, cam in cams:
        if name.lower() in cam.lower():
            return idx
    sys.exit(f'Camera "{name}" not found. Available: {cams}')


def center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", nargs="?", type=int, choices=range(0, 6), help="number of fingers held up (0-5); asked interactively if omitted")
    ap.add_argument("--list", action="store_true", help="list cameras and exit")
    ap.add_argument("--camera", default=CAMERA_NAME, help=f'camera name (default "{CAMERA_NAME}")')
    ap.add_argument("--fps", type=float, default=10, help="saved images per second (default 10)")
    ap.add_argument("--duration", type=float, default=30, help="seconds to capture (default 30; 0 = until quit)")
    ap.add_argument("--countdown", type=int, default=3, help="seconds before starting (default 3)")
    ap.add_argument("--size", type=int, default=128, help="output square size in px (default 128)")
    ap.add_argument("--gray", action="store_true", help="save grayscale images")
    ap.add_argument("--mirror", action="store_true", help="flip horizontally (selfie view) before saving")
    ap.add_argument("--min-sharpness", type=float, default=30.0, help="reject frames with Laplacian variance below this")
    ap.add_argument("--min-diff", type=float, default=0.3, help="skip frames whose mean abs difference from the last saved frame is below this (0 = off)")
    ap.add_argument("--out", type=Path, default=HERE / "dataset", help="output root (default ./dataset)")
    args = ap.parse_args()

    if args.list:
        for i, n in list_cameras():
            print(f"[{i}] {n}")
        return
    if args.label is None:
        while True:
            ans = input("How many fingers are being held up (0-5)? ").strip()
            if ans.isdigit() and 0 <= int(ans) <= 5:
                args.label = int(ans)
                break
            print("Please enter a whole number from 0 to 5.")

    idx = find_camera(args.camera)
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        sys.exit("Could not open camera. Check System Settings > Privacy & Security > Camera for your terminal/VS Code.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f'Using camera [{idx}] "{args.camera}", {int(cap.get(3))}x{int(cap.get(4))}')

    out_dir = args.out / str(args.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    win = "capture (SPACE pause, q quit)"

    def grab():
        ok, frame = cap.read()
        if not ok:
            sys.exit("Camera read failed.")
        return cv2.flip(frame, 1) if args.mirror else frame

    def show(frame, text, color=(0, 255, 0)):
        view = center_square(frame).copy()
        cv2.putText(view, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.imshow(win, view)
        return cv2.waitKey(1) & 0xFF

    # Countdown (keeps pulling frames so the camera auto-exposure settles)
    end = time.time() + args.countdown
    while time.time() < end:
        if show(grab(), f"Get ready: {end - time.time():.0f}+  label={args.label}", (0, 200, 255)) in (27, ord("q")):
            return

    interval = 1.0 / args.fps
    next_t = time.time()
    start = time.time()
    saved = rej_blur = rej_dup = 0
    last_small = None
    paused = False

    print(f"Capturing label {args.label} -> {out_dir}")
    while True:
        now = time.time()
        if args.duration and now - start >= args.duration:
            break
        frame = grab()  # always drain the buffer so frames stay fresh
        status = "PAUSED" if paused else f"REC label={args.label} saved={saved}"
        key = show(frame, status, (0, 0, 255) if paused else (0, 255, 0))
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
            next_t = time.time()
        if paused or now < next_t:
            continue
        next_t += interval
        if next_t < now:  # fell behind; don't burst
            next_t = now + interval

        sq = center_square(frame)
        gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < args.min_sharpness:
            rej_blur += 1
            continue
        img = cv2.resize(gray if args.gray else sq, (args.size, args.size), interpolation=cv2.INTER_AREA)

        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        if args.min_diff and last_small is not None and np.abs(small - last_small).mean() < args.min_diff:
            rej_dup += 1
            continue
        last_small = small

        cv2.imwrite(str(out_dir / f"{args.label}_{session}_{saved:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. saved={saved}  rejected_blurry={rej_blur}  rejected_duplicate={rej_dup}")
    print(f"Total images for label {args.label}: {len(list(out_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
