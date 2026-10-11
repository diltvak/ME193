"""Detect forward racket-style arm swings and prompt the user to swing at set times.

Combines the webcam pose tracking from pose_detection.py (wrist moving fast and toward the
camera, relative to the shoulder) with the LEGO Double Motor IMU from imu_test.py (gyro
rate; hold or strap the hub on the swinging arm). If the hub can't connect, or --no-imu
is passed, detection falls back to the camera alone.

A "SWING!" prompt with a shrinking timer bar is overlaid on the video at each cue time.
Swinging inside the window counts as a hit.

Run: python3 swing.py [--arm right|left] [--times 5,10,15] [--no-imu]  (q or ESC to quit)
Without --times, cues come at random intervals.
"""

import argparse
import math
import random
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from pose_detection import MODEL_PATH, PL, ensure_model

# ---- Configuration ----
MOTOR_CARD_SERIAL = "1151"
ANGLE_SCALE = 0.1         # the hub reports gyro rates in tenths of a degree/s

MIN_VISIBILITY = 0.5
SMOOTHING = 0.7           # EMA weight on the newest velocity sample (1 = no smoothing)
SPEED_THRESHOLD = 1.2     # wrist speed relative to shoulder, in shoulder-widths/s
FORWARD_THRESHOLD = 0.3   # component of that speed toward the camera, shoulder-widths/s
GYRO_THRESHOLD = 150.0    # IMU angular rate magnitude, deg/s
IMU_HOLD_S = 0.5          # how long an IMU spike keeps confirming a camera swing
SWING_COOLDOWN_S = 0.8    # ignore new detections right after a swing

GET_READY_S = 1.0         # "get ready" lead-in before each swing prompt
SWING_WINDOW_S = 1.5      # time allowed to swing once prompted
RESULT_S = 1.0            # how long HIT / MISS stays on screen
FIRST_CUE_S = 3.0
RANDOM_GAP_S = (3.0, 6.0)  # gap between cues when --times isn't given

GREEN, RED, YELLOW, WHITE = (0, 220, 0), (0, 0, 230), (0, 220, 255), (255, 255, 255)


class SwingDetector:
    """Tracks one wrist relative to its shoulder and flags fast forward motion."""

    def __init__(self, arm):
        self.wrist = PL[f"{arm.upper()}_WRIST"].value
        self.shoulder = PL[f"{arm.upper()}_SHOULDER"].value
        self.other_shoulder = PL["LEFT_SHOULDER" if arm == "right" else "RIGHT_SHOULDER"].value
        self.prev = None          # (time, relative wrist position in shoulder-widths)
        self.speed = 0.0
        self.forward = 0.0
        self.last_swing = -math.inf
        self.last_imu_spike = -math.inf

    def update(self, landmarks, now, gyro_dps):
        """Feed one frame. Returns True if a swing was just detected.

        gyro_dps is the IMU rate magnitude, or None when no IMU is in use.
        """
        if gyro_dps is not None and gyro_dps > GYRO_THRESHOLD:
            self.last_imu_spike = now

        w, s, o = (landmarks[i] for i in (self.wrist, self.shoulder, self.other_shoulder))
        if min(w.visibility, s.visibility, o.visibility) < MIN_VISIBILITY:
            self.prev = None
            return False

        # MediaPipe z shares x's scale, so everything is expressed in shoulder-widths.
        width = math.dist((s.x, s.y, s.z), (o.x, o.y, o.z)) or 1e-6
        rel = ((w.x - s.x) / width, (w.y - s.y) / width, (w.z - s.z) / width)
        prev, self.prev = self.prev, (now, rel)
        if prev is None or now <= prev[0]:
            return False

        dt = now - prev[0]
        vel = [(a - b) / dt for a, b in zip(rel, prev[1])]
        speed = math.sqrt(sum(v * v for v in vel))
        forward = -vel[2]   # smaller z = closer to the camera
        self.speed += SMOOTHING * (speed - self.speed)
        self.forward += SMOOTHING * (forward - self.forward)

        camera_swing = self.speed > SPEED_THRESHOLD and self.forward > FORWARD_THRESHOLD
        imu_ok = gyro_dps is None or now - self.last_imu_spike < IMU_HOLD_S
        if camera_swing and imu_ok and now - self.last_swing > SWING_COOLDOWN_S:
            self.last_swing = now
            return True
        return False


class Prompter:
    """Schedules swing cues and scores whether a swing lands inside each window."""

    def __init__(self, cue_times):
        self.cue_times = sorted(cue_times) if cue_times else None
        self.state = "wait"       # wait -> ready -> swing -> result -> wait
        self.state_start = 0.0
        self.next_cue = 0.0       # time the swing window opens
        self.hit = False
        self.hits = self.misses = 0

    def start(self, now):
        self.state_start = now
        self.next_cue = self._pop_next_cue(now, FIRST_CUE_S)

    def _pop_next_cue(self, now, default_gap):
        if self.cue_times is None:
            return now + default_gap
        return self.cue_times.pop(0) if self.cue_times else math.inf

    def finished(self):
        return self.cue_times is not None and not self.cue_times and self.state == "wait" \
            and self.next_cue == math.inf

    def update(self, now, swung):
        if self.state == "wait" and now >= self.next_cue - GET_READY_S:
            self._go("ready", now)
        elif self.state == "ready" and now >= self.next_cue:
            self._go("swing", now)
            self.hit = False
        elif self.state == "swing":
            if swung:
                self.hit = True
                self.hits += 1
                self._go("result", now)
            elif now - self.state_start >= SWING_WINDOW_S:
                self.misses += 1
                self._go("result", now)
        elif self.state == "result" and now - self.state_start >= RESULT_S:
            self._go("wait", now)
            self.next_cue = self._pop_next_cue(now, random.uniform(*RANDOM_GAP_S))

    def _go(self, state, now):
        self.state, self.state_start = state, now


def put_centered(frame, text, y, scale, color, thickness=3):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = (frame.shape[1] - size[0]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_prompt(frame, prompter, now):
    h, w = frame.shape[:2]
    elapsed = now - prompter.state_start
    if prompter.state == "ready":
        put_centered(frame, "Get ready...", h // 2, 1.6, YELLOW)
    elif prompter.state == "swing":
        # Pulsing green border, big SWING! text, and a bar that drains over the window.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), GREEN, 40 if int(elapsed * 8) % 2 else 20)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        put_centered(frame, "SWING!", h // 2, 3.0, GREEN, 6)
        remaining = max(0.0, 1 - elapsed / SWING_WINDOW_S)
        cv2.rectangle(frame, (w // 4, h - 60), (w // 4 + int(w / 2 * remaining), h - 40), GREEN, -1)
        cv2.rectangle(frame, (w // 4, h - 60), (3 * w // 4, h - 40), WHITE, 2)
    elif prompter.state == "result":
        overlay = frame.copy()
        color = GREEN if prompter.hit else RED
        cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        put_centered(frame, "HIT!" if prompter.hit else "MISS", h // 2, 3.0, color, 6)


def draw_arm(frame, landmarks, arm, swung):
    h, w = frame.shape[:2]
    pts = []
    for name in ("SHOULDER", "ELBOW", "WRIST"):
        lm = landmarks[PL[f"{arm.upper()}_{name}"].value]
        if lm.visibility < MIN_VISIBILITY:
            return
        pts.append((int((1 - lm.x) * w), int(lm.y * h)))   # x mirrored to match the flipped view
    color = GREEN if swung else (0, 128, 255)
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, color, 4)
    for p in pts:
        cv2.circle(frame, p, 8, color, -1)
        cv2.circle(frame, p, 8, WHITE, 1)


def draw_hud(frame, prompter, detector, gyro_dps, swing_flash):
    lines = [f"Hits {prompter.hits}   Misses {prompter.misses}",
             f"Wrist speed {detector.speed:4.1f} (>{SPEED_THRESHOLD})  forward {detector.forward:4.1f} (>{FORWARD_THRESHOLD})",
             f"Gyro {gyro_dps:5.0f} deg/s (>{GYRO_THRESHOLD:.0f})" if gyro_dps is not None else "IMU: off (camera only)"]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    if swing_flash:
        cv2.putText(frame, "swing detected", (10, 25 + 24 * len(lines)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, GREEN, 2, cv2.LINE_AA)


def connect_imu():
    """Connect to the Double Motor; returns it, or None if it can't be reached."""
    import legoeducation as le
    motor = le.DoubleMotor()
    print("Connecting to Double Motor...")
    motor.connect(card_color=le.LEGO_COLOR_ORANGE, card_serial=MOTOR_CARD_SERIAL)
    if not motor.connected:
        print("Could not connect to the Double Motor; continuing with camera only.")
        return None
    return motor


def gyro_magnitude(motor):
    imu = motor.imu_device
    return ANGLE_SCALE * math.sqrt(imu.gyroscopeX ** 2 + imu.gyroscopeY ** 2 + imu.gyroscopeZ ** 2)


def main():
    parser = argparse.ArgumentParser(description="Detect forward arm swings and prompt the user to swing.")
    parser.add_argument("--arm", choices=("right", "left"), default="right", help="swinging arm")
    parser.add_argument("--times", type=lambda s: [float(t) for t in s.split(",")],
                        help="comma-separated cue times in seconds after start, e.g. 5,10,15")
    parser.add_argument("--no-imu", action="store_true", help="use the camera only")
    args = parser.parse_args()

    ensure_model()
    motor = None if args.no_imu else connect_imu()
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

    detector = SwingDetector(args.arm)
    start = time.monotonic()
    # Cue times are relative to start; the prompter works in absolute monotonic time.
    prompter = Prompter([start + t for t in args.times] if args.times else None)
    prompter.start(start)
    swing_flash_until = 0.0
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while not prompter.finished():
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                # Detect on the unflipped frame so MediaPipe's left/right labels are correct,
                # then mirror only for display.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))
                frame = cv2.flip(frame, 1)

                gyro_dps = gyro_magnitude(motor) if motor else None
                swung = False
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    swung = detector.update(landmarks, now, gyro_dps)
                    if swung:
                        swing_flash_until = now + 0.5
                    draw_arm(frame, landmarks, args.arm, now < swing_flash_until)
                else:
                    detector.prev = None

                prompter.update(now, swung)
                draw_prompt(frame, prompter, now)
                draw_hud(frame, prompter, detector, gyro_dps, now < swing_flash_until)

                cv2.imshow("Swing trainer (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if motor and motor.connected:
            motor.disconnect()
    print(f"Hits: {prompter.hits}  Misses: {prompter.misses}")


if __name__ == "__main__":
    main()
