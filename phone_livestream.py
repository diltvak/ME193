import cv2

def live_iphone_stream():
    # 0 is typically the MacBook's built-in FaceTime camera.
    # 1 is usually the iPhone when Continuity Camera is active.
    # You may need to change this index to 0 or 2 depending on connected peripherals.
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera stream. Verify camera index and macOS terminal permissions.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # --- INSERT VIDEO ANALYSIS LOGIC HERE ---
        # e.g., HSV thresholding, contour detection, or centroid calculations
        
        # Display the live stream
        cv2.imshow('iPhone Analysis Feed', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_iphone_stream()