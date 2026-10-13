"""Real-time facial expression recognition overlaid on the laptop webcam.

Uses DeepFace (open source) for face detection + emotion classification and OpenCV for
capture/drawing. Inference runs in a background thread so the video stays smooth.
Press 'q' or ESC in the video window to quit.
"""

import os
import threading
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # DeepFace models need Keras 2 (tf-keras)

import cv2
import numpy as np
from deepface import DeepFace

CAMERA_INDEX = 0
DETECTOR = "opencv"  # fast; alternatives: "ssd", "mediapipe", "retinaface" (slower, more accurate)
MIN_FACE_CONFIDENCE = 0.5
SMOOTHING = 0.6  # 0 = no smoothing, closer to 1 = slower, steadier label changes

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
COLORS = {  # BGR
    "angry": (0, 0, 255),
    "disgust": (0, 140, 100),
    "fear": (200, 0, 160),
    "happy": (0, 220, 120),
    "sad": (255, 128, 0),
    "surprise": (0, 215, 255),
    "neutral": (200, 200, 200),
}


class EmotionWorker(threading.Thread):
    """Continuously analyzes the most recent frame and publishes the latest results."""

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._frame = None
        self._results = []  # list of (box, smoothed scores dict)
        self._running = True

    def submit(self, frame):
        with self._lock:
            self._frame = frame

    def results(self):
        with self._lock:
            return list(self._results)

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                faces = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend=DETECTOR,
                    enforce_detection=False,
                    silent=True,
                )
            except Exception as exc:  # keep the loop alive on transient failures
                print(f"analysis error: {exc}")
                continue
            if isinstance(faces, dict):
                faces = [faces]
            faces = [f for f in faces if f.get("face_confidence", 1) >= MIN_FACE_CONFIDENCE]
            with self._lock:
                self._results = self._smooth(faces, self._results)

    @staticmethod
    def _smooth(faces, previous):
        """Blend each face's scores with the nearest previous face to reduce flicker."""
        out = []
        for f in faces:
            r = f["region"]
            box = (r["x"], r["y"], r["w"], r["h"])
            scores = {e: float(f["emotion"][e]) for e in EMOTIONS}
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            for pbox, pscores in previous:
                px, py = pbox[0] + pbox[2] / 2, pbox[1] + pbox[3] / 2
                if abs(cx - px) < pbox[2] and abs(cy - py) < pbox[3]:
                    scores = {e: SMOOTHING * pscores[e] + (1 - SMOOTHING) * scores[e] for e in EMOTIONS}
                    break
            out.append((box, scores))
        return out


def draw_overlay(frame, results):
    for (x, y, w, h), scores in results:
        top = max(scores, key=scores.get)
        color = COLORS[top]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{top} {scores[top]:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ly = max(y, th + 10)
        cv2.rectangle(frame, (x, ly - th - 10), (x + tw + 8, ly), color, -1)
        cv2.putText(frame, label, (x + 4, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Probability bars for every emotion beside the face (or inside the frame edge).
        bar_x = x + w + 10 if x + w + 190 < frame.shape[1] else max(x - 190, 0)
        for i, emo in enumerate(EMOTIONS):
            by = y + i * 20
            cv2.rectangle(frame, (bar_x, by), (bar_x + 100, by + 14), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, by), (bar_x + int(scores[emo]), by + 14), COLORS[emo], -1)
            cv2.putText(frame, emo, (bar_x + 106, by + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}")

    print("Loading emotion model (first run downloads weights)...")
    DeepFace.analyze(
        np.zeros((96, 96, 3), np.uint8), actions=["emotion"], enforce_detection=False, silent=True
    )

    worker = EmotionWorker()
    worker.start()
    fps, last = 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            worker.submit(frame.copy())
            draw_overlay(frame, worker.results())

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            cv2.putText(frame, f"{fps:.0f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Facial Expressions", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
