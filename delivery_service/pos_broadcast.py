"""Track the Lego minifigure on the webcam and broadcast its XY position over MQTT.

Same detector as webcam_test.py, but each frame the most confident detection's
position is published to the MQTT topic "position" as JSON, e.g.
    {"x": 0.125, "y": -0.5}
Coordinates are normalized and relative to the very center of the frame, so
they don't depend on the camera's resolution: (0, 0) is the center, x = -1 / 1
is the left / right edge, and y = -1 / 1 is the bottom / top edge.

Usage:  python pos_broadcast.py [--weights best.pt] [--camera 1] [--conf 0.4]
        python pos_broadcast.py --list      # show which camera indexes work
Press q or ESC to quit.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from webcam_test import list_cameras, open_camera

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "2026-09-22"))
from mqttlib import MQTTClient  # noqa: E402

TOPIC = "position"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "best.pt"))
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

    client = MQTTClient()
    client.connect()
    print(f"Connected to {client.broker}, publishing to topic '{TOPIC}'")

    failed_reads = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Tolerate a few empty frames while the camera warms up.
                failed_reads += 1
                if failed_reads > 30:
                    raise SystemExit(f"Camera {args.camera} isn't returning frames. "
                                     "Try another index (python pos_broadcast.py --list).")
                continue
            failed_reads = 0

            result = model(frame, conf=args.conf, verbose=False)[0]
            h, w = frame.shape[:2]
            mid_x, mid_y = w // 2, h // 2

            # Crosshair marking the origin of the broadcast coordinates.
            cv2.drawMarker(frame, (mid_x, mid_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)

            if len(result.boxes):
                box = result.boxes[int(result.boxes.conf.argmax())]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf = float(box.conf[0])
                # Scale so the frame edges are +/-1; y flipped so up is positive.
                x = round((cx - mid_x) / (w / 2), 3)
                y = round((mid_y - cy) / (h / 2), 3)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.line(frame, (mid_x, mid_y), (cx, cy), (255, 0, 0), 1)
                label = f"Lego {conf:.2f}  pos=({x:+.2f},{y:+.2f})"
                cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                message = json.dumps({"x": x, "y": y})
                client.publish(TOPIC, message)
                print(f"[{TOPIC}] {message}")

            cv2.imshow("Lego minifigure position broadcaster", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        client.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
