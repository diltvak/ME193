import cv2
import numpy as np
import time
import legoeducation as le

# =====================================================================
# LEGO CONNECTION CARD - set these to match YOUR Double Motor so this
# script only ever connects to your robot, not any other nearby unit.
# Color options (see constants.md): LEGO_COLOR_GREEN, LEGO_COLOR_BLUE,
# LEGO_COLOR_RED, LEGO_COLOR_ORANGE, LEGO_COLOR_YELLOW, LEGO_COLOR_AZURE,
# LEGO_COLOR_PURPLE, LEGO_COLOR_MAGENTA
# =====================================================================
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = "6235"       # string, e.g. "0049" (keep leading zeros as a string)
# =====================================================================

# --- Proportional controller tuning ---
KP = 0.12                 # motor speed (%) per pixel of horizontal error
MAX_DRIVE_SPEED = 40      # cap on commanded drive speed (%), -100..100
DEADBAND_PX = 15          # |dx| below this counts as "centered" -> stop driving
COMMAND_PERIOD_S = 0.05   # minimum time between BLE motor commands (~20 Hz)

# Flip this if the robot drives the wrong way relative to the tag once tested.
REVERSE_DRIVE = False


def compute_drive_speed(dx):
    """Proportional controller: pixel error -> forward/reverse drive speed.

    The robot only drives straight (forward/backward), so both motors
    receive the same speed. Positive dx (tag right of center) drives
    forward; negative dx drives in reverse; within the deadband it stops.
    """
    if abs(dx) < DEADBAND_PX:
        return 0

    speed = KP * dx
    speed = max(-MAX_DRIVE_SPEED, min(MAX_DRIVE_SPEED, speed))
    if REVERSE_DRIVE:
        speed = -speed

    return speed


def run_continuous_apriltag_tracker(camera_index=1, target_tag_id=None):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        detect_fn = detector.detectMarkers
    else:
        detect_fn = lambda img: cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    # --- Connect to your specific LEGO Double Motor over Bluetooth LE ---
    doublemotor = le.DoubleMotor()

    print("Connecting to LEGO Double Motor...")
    doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not doublemotor.connected:
        print("Error connecting to Double Motor.")
        return

    cap = cv2.VideoCapture(camera_index)

    print("Starting continuous feed. Press 'q' in the video window or Ctrl+C in terminal to exit.")

    last_command_time = 0.0
    last_sent_speed = None

    try:
        while True:
            if not cap.isOpened():
                print("Camera feed disconnected. Attempting to reconnect...")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(camera_index)
                continue

            ret, frame = cap.read()

            if not ret or frame is None:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                continue

            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_fn(gray)

            cv2.line(frame, (center_x, 0), (center_x, h), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

            dx = None

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, corner_group in enumerate(corners):
                    pts = corner_group.reshape((4, 2))
                    tag_cx = int(np.mean(pts[:, 0]))
                    tag_cy = int(np.mean(pts[:, 1]))
                    tag_id = int(np.ravel(ids[i])[0])

                    tag_dx = tag_cx - center_x
                    tag_dy = tag_cy - center_y

                    h_pos = f"Right: {tag_dx}px" if tag_dx > 0 else f"Left: {abs(tag_dx)}px" if tag_dx < 0 else "Centered"

                    cv2.circle(frame, (tag_cx, tag_cy), 5, (0, 255, 0), -1)
                    cv2.line(frame, (center_x, tag_cy), (tag_cx, tag_cy), (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame, f"ID: {tag_id}", (tag_cx - 40, tag_cy - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.putText(frame, f"dx: {tag_dx}px ({h_pos})", (tag_cx - 40, tag_cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                    # Track the requested tag id, or the first tag seen if none specified.
                    if target_tag_id is None or tag_id == target_tag_id:
                        if dx is None:
                            dx = tag_dx

            tag_count = len(ids) if ids is not None else 0
            cv2.putText(frame, f"Tags Detected: {tag_count}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- Proportional control loop ---
            if dx is not None:
                drive_speed = compute_drive_speed(dx)
            else:
                drive_speed = 0

            now = time.time()
            speed_changed = drive_speed != last_sent_speed
            if speed_changed and (now - last_command_time) >= COMMAND_PERIOD_S:
                doublemotor.movement_move_tank(drive_speed, drive_speed, blocking=False)
                last_sent_speed = drive_speed
                last_command_time = now

            cv2.putText(frame, f"Drive: {drive_speed:.0f}%", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            cv2.imshow("AprilTag Tracking (36h11)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        doublemotor.movement_stop()
        doublemotor.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # camera_index=1 uses the iPhone via Continuity Camera (0 is usually the
    # Mac's built-in FaceTime camera). Adjust if your setup differs.
    run_continuous_apriltag_tracker(camera_index=1)
