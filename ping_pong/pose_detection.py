"""Real-time arm / leg / wrist detection from the laptop webcam.

Uses MediaPipe's PoseLandmarker (33 body landmarks) and OpenCV for capture/drawing.
Press 'q' or ESC in the video window to quit.
"""

import os
import ssl
import time
import urllib.request

import certifi
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")

PL = vision.PoseLandmark

# Landmarks we report, grouped by limb.
LIMBS = {
    "left_arm": [PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST],
    "right_arm": [PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST],
    "left_leg": [PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE],
    "right_leg": [PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE],
}
COLORS = {  # BGR
    "left_arm": (255, 128, 0),
    "right_arm": (0, 128, 255),
    "left_leg": (255, 0, 128),
    "right_leg": (0, 220, 120),
}
MIN_VISIBILITY = 0.5


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model to {MODEL_PATH} ...")
        # python.org builds of Python on macOS lack system certs, so use certifi's bundle.
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(MODEL_PATH, "wb") as f:
            f.write(r.read())


def get_limb_positions(landmarks, width, height):
    """Return {landmark_name: (x_px, y_px, z, visibility)} for arms, wrists and legs."""
    out = {}
    for joints in LIMBS.values():
        for j in joints:
            lm = landmarks[j.value]
            out[j.name.lower()] = (int(lm.x * width), int(lm.y * height), lm.z, lm.visibility)
    return out


def draw_limbs(frame, positions):
    for limb, joints in LIMBS.items():
        pts = []
        for j in joints:
            x, y, _, vis = positions[j.name.lower()]
            pts.append((x, y) if vis >= MIN_VISIBILITY else None)
        color = COLORS[limb]
        for a, b in zip(pts, pts[1:]):
            if a and b:
                cv2.line(frame, a, b, color, 3)
        for j, p in zip(joints, pts):
            if p:
                radius = 10 if "WRIST" in j.name else 6
                cv2.circle(frame, p, radius, color, -1)
                cv2.circle(frame, p, radius, (255, 255, 255), 1)
                if "WRIST" in j.name:
                    cv2.putText(frame, f"{j.name.lower()} ({p[0]},{p[1]})", (p[0] + 12, p[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (check macOS camera permission for your terminal/IDE).")

    start = time.monotonic()
    last_print = 0.0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.monotonic() - start) * 1000))

            if result.pose_landmarks:
                positions = get_limb_positions(result.pose_landmarks[0], w, h)
                draw_limbs(frame, positions)
                now = time.monotonic()
                if now - last_print > 0.5:  # print twice a second
                    last_print = now
                    print({k: v[:2] for k, v in positions.items() if "wrist" in k})

            cv2.imshow("Pose detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
