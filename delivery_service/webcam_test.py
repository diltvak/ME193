"""Run the trained model on the webcam and show the Lego minifigure's location.

Usage:  python webcam_test.py [--weights runs/lego_dude/weights/best.pt]
        [--camera 1] [--conf 0.4]
        python webcam_test.py --list      # show which camera indexes work
Press q or ESC to quit.

On a Mac, an iPhone nearby can show up as a Continuity Camera and take
index 0. Run with --list, then pick the index of the built-in webcam.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def open_camera(index):
    # AVFoundation is the native macOS backend; CAP_ANY elsewhere.
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def list_cameras(max_index=5):
    print("Probing cameras (a preview window opens for each one that works)...")
    for i in range(max_index):
        cap = open_camera(i)
        ok, frame = False, None
        if cap.isOpened():
            for _ in range(20):  # cameras often return empty frames at first
                ok, frame = cap.read()
                if ok:
                    break
        if ok:
            h, w = frame.shape[:2]
            print(f"  camera {i}: works ({w}x{h}) - press any key in the window for the next one")
            cv2.imshow(f"camera {i}", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"  camera {i}: not available")
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "delivery_service"/"best.pt"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--list", action="store_true", help="list working camera indexes and exit")
    args = ap.parse_args()

    if args.list:
        list_cameras()
        return

    model = YOLO(args.weights)
    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --list.")

    failed_reads = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            # Tolerate a few empty frames while the camera warms up.
            failed_reads += 1
            if failed_reads > 30:
                raise SystemExit(f"Camera {args.camera} isn't returning frames. "
                                 "Try another index (python webcam_test.py --list).")
            continue
        failed_reads = 0

        result = model(frame, conf=args.conf, verbose=False)[0]
        h, w = frame.shape[:2]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            label = f"Lego {conf:.2f}  center=({cx},{cy})  norm=({cx / w:.2f},{cy / h:.2f})"
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"center=({cx},{cy}) bbox=({x1},{y1},{x2},{y2}) conf={conf:.2f}")

        cv2.imshow("Lego minifigure detector", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
