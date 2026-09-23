import cv2
import numpy as np
import time

def run_continuous_apriltag_tracker(camera_index=1):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        detect_fn = detector.detectMarkers
    else:
        detect_fn = lambda img: cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    cap = cv2.VideoCapture(camera_index)

    print("Starting continuous feed. Press 'q' in the video window or Ctrl+C in terminal to exit.")

    try:
        while True:
            # If camera isn't open or got disconnected, attempt to reconnect
            if not cap.isOpened():
                print("Camera feed disconnected. Attempting to reconnect...")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(camera_index)
                continue

            ret, frame = cap.read()

            # Handle momentary frame drops or screen lock pauses without exiting
            if not ret or frame is None:
                # Give the camera buffer a moment to recover
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                continue

            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_fn(gray)

            # Draw center reference line
            cv2.line(frame, (center_x, 0), (center_x, h), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, corner_group in enumerate(corners):
                    pts = corner_group.reshape((4, 2))
                    tag_cx = int(np.mean(pts[:, 0]))
                    tag_cy = int(np.mean(pts[:, 1]))
                    tag_id = int(np.ravel(ids[i])[0])

                    dx = tag_cx - center_x
                    dy = tag_cy - center_y

                    h_pos = f"Right: {dx}px" if dx > 0 else f"Left: {abs(dx)}px" if dx < 0 else "Centered"
                    
                    cv2.circle(frame, (tag_cx, tag_cy), 5, (0, 255, 0), -1)
                    cv2.line(frame, (center_x, tag_cy), (tag_cx, tag_cy), (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.putText(frame, f"ID: {tag_id}", (tag_cx - 40, tag_cy - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.putText(frame, f"dx: {dx}px ({h_pos})", (tag_cx - 40, tag_cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            tag_count = len(ids) if ids is not None else 0
            cv2.putText(frame, f"Tags Detected: {tag_count}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("AprilTag Tracking (36h11)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_continuous_apriltag_tracker()