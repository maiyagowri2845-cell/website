from flask import Flask, Response
import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO

from ensemble import ensemble_hard_voting
from config import (
    ENSEMBLE_VOTING_THRESHOLD, OD_IOU_THRESHOLD,
    OD_MODEL_PATH_B, OD_MODEL_PATH_C,
    OD_TO_CLF_MAP_B, OD_TO_CLF_MAP_C
)

# ------------------------------
# SETTINGS — TUNE FOR SPEED
# ------------------------------
YOLO_FRAME_SKIP = 6      # run YOLO every 6th frame
RESIZE_WIDTH = 480       # reduce size for faster inference

# ------------------------------
# Load Models
# ------------------------------
print("Loading YOLO models...")
model_B = YOLO(OD_MODEL_PATH_B)
model_C = YOLO(OD_MODEL_PATH_C)

NAMES_B = list(model_B.names.values())
NAMES_C = list(model_C.names.values())

# ------------------------------
# Global Frame Buffer
# ------------------------------
latest_frame = None
lock = threading.Lock()

# ------------------------------
# CAMERA THREAD (FAST)
# ------------------------------
def camera_thread():
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize for speed
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(RESIZE_WIDTH * 0.75)))

        with lock:
            latest_frame = frame.copy()

# ------------------------------
# YOLO DETECTION THREAD (SLOWER)
# ------------------------------
detections_B = []
detections_C = []

def run_od(model, frame, names):
    res = model(frame, conf=0.25, iou=OD_IOU_THRESHOLD,
                imgsz=416, verbose=False, device="cpu")

    labels, confs, boxes = [], [], []

    if len(res) and len(res[0].boxes):
        r = res[0].boxes
        xy = r.xyxy.cpu().numpy().astype(int)
        cf = r.conf.cpu().numpy()
        cl = r.cls.cpu().numpy().astype(int)

        for b, c, k in zip(xy, cf, cl):
            labels.append(names[k])
            confs.append(float(c))
            boxes.append(tuple(b))

    return labels, confs, boxes


def detection_thread():
    global detections_B, detections_C

    frame_count = 0

    while True:
        if latest_frame is None:
            continue

        frame_count += 1

        # Skip frames to improve speed
        if frame_count % YOLO_FRAME_SKIP != 0:
            time.sleep(0.005)
            continue

        with lock:
            frame = latest_frame.copy()

        # Run YOLO models
        labels_B, conf_B, boxes_B = run_od(model_B, frame, NAMES_B)
        labels_C, conf_C, boxes_C = run_od(model_C, frame, NAMES_C)

        # Save detections
        detections_B = (labels_B, conf_B, boxes_B)
        detections_C = (labels_C, conf_C, boxes_C)

# ------------------------------
# FLASK STREAM
# ------------------------------
app = Flask(__name__)

def generate_stream():
    global latest_frame

    while True:
        if latest_frame is None:
            continue

        with lock:
            frame = latest_frame.copy()

        # Draw latest YOLO detections
        labels_B, conf_B, boxes_B = detections_B
        labels_C, conf_C, boxes_C = detections_C

        for lbl, conf, box in zip(labels_B, conf_B, boxes_B):
            x1,y1,x2,y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{lbl} {conf:.2f}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        for lbl, conf, box in zip(labels_C, conf_C, boxes_C):
            x1,y1,x2,y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
            cv2.putText(frame, f"{lbl} {conf:.2f}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

@app.route("/")
def index():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------------------------
# START THREADS & SERVER
# ------------------------------
if __name__ == "__main__":
    print("Starting FAST live stream...")

    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=detection_thread, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=False)

