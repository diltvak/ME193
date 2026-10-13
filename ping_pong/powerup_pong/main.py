"""Ping pong against a virtual opponent, played with real arm swings, plus minifigure powerups.

Same game as first_pong.py (split into modules), with two once-per-game powerups triggered by
showing a LEGO minifigure to the camera (detected by the YOLO model in weights.pt):

  Yellow minifig  BRICK WALL  a wall rises at the end of the table and returns the ball for you
  Green minifig   ICE STORM   an ice storm slows the ball for the next 3 returns

Setup screen keys:  1/2/3 difficulty   P points to win   A swap arm   H haptics   M powerups
                    W warm-up   ENTER/SPACE start   Q quit
Any time:           F fullscreen   V game-only view (camera becomes an inset)   + / - bigger / smaller
                    (the window can also be resized by dragging its edge)
In a game or warm-up: ESC returns to setup, Q quits. Y / G fire the yellow / green powerup from
the keyboard, for testing without a minifig. In warm-up, powerups recharge after use.

Run from ping_pong/:  python -m powerup_pong [--arm right|left] [--difficulty easy|medium|hard]
                                              [--points 7] [--width 560] [--no-imu] [--no-powerups]
"""

import argparse
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ping_pong/, for swing.py etc.

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from pose_detection import MODEL_PATH, ensure_model
from swing import GYRO_THRESHOLD, SPEED_THRESHOLD, SwingDetector, connect_imu, draw_arm, gyro_magnitude

from powerup_pong.config import DIFFICULTIES, POINT_CHOICES, POWER_WINDOW_S, PRACTICE, WEIGHTS_PATH
from powerup_pong.detection import MinifigDetector
from powerup_pong.game import Game
from powerup_pong.haptics import Haptics
from powerup_pong.opponent import Opponent
from powerup_pong.paddle import Paddle
from powerup_pong.powerups import make_powerups
from powerup_pong.render import draw_bottom_info, draw_menu, draw_minifigs, draw_top
from powerup_pong.view import View


def main():
    parser = argparse.ArgumentParser(description="Ping pong played with arm swings, with minifigure powerups.")
    parser.add_argument("--arm", choices=("right", "left"), default="right", help="swinging arm")
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="medium")
    parser.add_argument("--points", type=int, default=7, help="points needed to win")
    parser.add_argument("--width", type=int, default=560, help="width of each half of the window, in pixels")
    parser.add_argument("--no-imu", action="store_true", help="use the camera only")
    parser.add_argument("--no-powerups", action="store_true", help="start with powerups off (toggle with M)")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="YOLO weights for minifigure detection")
    args = parser.parse_args()

    ensure_model()
    motor = None if args.no_imu else connect_imu()
    minifig = MinifigDetector(args.weights)    # loads in the background while the menu is up
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
    view = View("Ping pong (q to quit)", args.width)

    arm, difficulty, points = args.arm, args.difficulty, args.points
    powerups_on = not args.no_powerups
    detector = SwingDetector(arm)
    paddle = Paddle()
    haptics = Haptics(motor)
    power_log = deque(maxlen=60)     # (time, swing power) samples
    mode, game, opponent = "menu", None, Opponent()
    flash_until = 0.0
    start = time.monotonic()

    def new_game(practice, now):
        cfg = PRACTICE if practice else DIFFICULTIES[difficulty]
        nonlocal opponent
        opponent = Opponent()
        return Game(cfg, points, practice, now, make_powerups() if powerups_on else [])

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                # Detect on the unflipped frame so left/right labels are right, then mirror for display.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now - start) * 1000))
                bottom = cv2.resize(cv2.flip(frame, 1), (view.pw, view.ph))
                if powerups_on:
                    minifig.submit(frame)

                # The motors' own vibration would read as a swing, so the gyro is ignored while they run.
                gyro_dps = (0.0 if haptics.busy else gyro_magnitude(motor)) if motor else None
                if not haptics.busy:
                    # Swing power: gyro rate, or camera wrist speed scaled to the same units without an IMU.
                    power_log.append((now, gyro_dps if gyro_dps is not None
                                      else detector.speed * GYRO_THRESHOLD / SPEED_THRESHOLD))
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                swung = False
                if landmarks:
                    swung = detector.update(landmarks, now, gyro_dps)
                    if swung:
                        flash_until = now + 0.5
                    draw_arm(bottom, landmarks, arm, now < flash_until)
                else:
                    detector.prev = None
                paddle.update(landmarks, arm)

                if mode == "menu":
                    draw_menu(view.top, difficulty, points, arm, paddle.tracked, motor is not None, gyro_dps or 0.0,
                              haptics.enabled, powerups_on, minifig.status)
                    minifig.poll()   # discard sightings made while in the menu
                else:
                    for name in minifig.poll():
                        game.powerups.handle_detection(game, name, now)
                    power = max((p for t, p in power_log if now - t <= POWER_WINDOW_S), default=0.0)
                    game.update(now, swung, paddle.xy, power)
                    if game.haptic:
                        haptics.play(game.haptic)
                        game.haptic = None
                    draw_top(view.top, view.court, game, paddle, detector, opponent, gyro_dps, now, now < flash_until)
                draw_bottom_info(bottom, detector, gyro_dps, now < flash_until)
                if powerups_on:
                    draw_minifigs(bottom, minifig)

                view.show(view.compose(bottom))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or (key == 27 and mode == "menu"):
                    break
                if view.handle_key(key):
                    continue
                if key == 27:
                    mode = "menu"
                elif mode == "menu":
                    if key in (ord("1"), ord("2"), ord("3")):
                        difficulty = tuple(DIFFICULTIES)[key - ord("1")]
                    elif key == ord("p"):
                        points = POINT_CHOICES[(POINT_CHOICES.index(points) + 1) % len(POINT_CHOICES)
                                               if points in POINT_CHOICES else 0]
                    elif key == ord("a"):
                        arm = "left" if arm == "right" else "right"
                        detector = SwingDetector(arm)
                    elif key == ord("h") and motor:
                        haptics.enabled = not haptics.enabled
                    elif key == ord("m"):
                        powerups_on = not powerups_on
                    elif key == ord("w"):
                        mode, game = "warmup", new_game(True, now)
                    elif key in (13, 10, 32):
                        mode, game = "game", new_game(False, now)
                elif key in (13, 10, 32) and (mode == "warmup" or game.phase == "over"):
                    mode = "menu"
                elif key != 255:
                    game.powerups.handle_key(game, key, now)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        haptics.close()
        minifig.close()
        if motor and motor.connected:
            motor.disconnect()


if __name__ == "__main__":
    main()
