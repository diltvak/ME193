"""
Live webcam detection with a YOLO11 model (weights.pt from Roboflow).
 
Install:   pip install ultralytics opencv-python
Run:       python webcam_detect.py
Controls:  q = quit, s = save screenshot
"""
 
import time
import cv2
from ultralytics import YOLO
 
# ---- Settings you can change ----
WEIGHTS = "weights.pt"   # path to your downloaded file
CAMERA_INDEX = 0         # try 1 if you have an external/other camera
CONF = 0.4               # confidence threshold (0-1)
IMGSZ = 640              # match the size you trained at
DEVICE = None            # None = auto, "cpu", or 0 / "cuda" for GPU
# ---------------------------------
 
model = YOLO(WEIGHTS)
 
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera {CAMERA_INDEX}. Try a different CAMERA_INDEX "
        "or check that no other app is using the webcam."
    )
 
prev_time = time.time()
shot_count = 0
 
while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame from webcam.")
        break
 
    results = model(frame, conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)
    r = results[0]
    annotated = r.plot()  # frame with boxes and labels drawn
 
    # FPS counter
    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
 
    # Print detections to the console (class name + confidence)
    for box in r.boxes:
        name = r.names[int(box.cls)]
        print(f"{name}: {float(box.conf):.2f}")
 
    cv2.imshow("YOLO11 Webcam", annotated)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        filename = f"screenshot_{shot_count}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"Saved {filename}")
        shot_count += 1
 
cap.release()
cv2.destroyAllWindows()