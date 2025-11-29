from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
from ultralytics import YOLO
from config import (
    OD_IOU_THRESHOLD, OD_MODEL_PATH_B, OD_MODEL_PATH_C,
    OD_TO_CLF_MAP_B, OD_TO_CLF_MAP_C
)
from ensemble import ensemble_hard_voting

app = Flask(__name__)

# ----------- GLOBAL STATES -------------
running = False
frame_lock = threading.Lock()
output_frame = None

# ----------- LOAD YOLO MODELS -----------
print("Loading YOLO Models...")
modelB = YOLO(OD_MODEL_PATH_B)
modelC = YOLO(OD_MODEL_PATH_C)
namesB = list(modelB.names.values())
namesC = list(modelC.names.values())
print("Models Loaded.")

cam = cv2.VideoCapture(0)

# ----------- YOLO OD FUNCTION -----------
def run_od(model, frame, names):
    res = model(frame, conf=0.25, iou=OD_IOU_THRESHOLD, imgsz=320, verbose=False, device="cpu")
    labels, confs, boxes = [], [], []

    if len(res) and len(res[0].boxes):
        box = res[0].boxes
        xyxy = box.xyxy.cpu().numpy().astype(int)
        conf = box.conf.cpu().numpy()
        cls = box.cls.cpu().numpy().astype(int)

        for b, c, k in zip(xyxy, conf, cls):
            labels.append(names[k])
            confs.append(float(c))
            boxes.append(tuple(b))
    return labels, confs, boxes

# ----------- BACKGROUND DETECTION THREAD -----------
def detect_loop():
    global output_frame, running

    while True:
        if not running:
            time.sleep(0.1)
            continue

        ret, frame = cam.read()
        if not ret:
            continue

        # Run detection
        labelsB, confB, boxB = run_od(modelB, frame, namesB)
        labelsC, confC, boxC = run_od(modelC, frame, namesC)

        final_label, final_conf = ensemble_hard_voting(
            [(labelsB, confB, OD_TO_CLF_MAP_B),
             (labelsC, confC, OD_TO_CLF_MAP_C)]
        )

        # Draw boxes (fast method)
        for (labels, confs, boxes) in [(labelsB, confB, boxB), (labelsC, confC, boxC)]:
            for lab, cf, box in zip(labels, confs, boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{lab} {cf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.putText(frame, final_label, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,255), 2)

        with frame_lock:
            output_frame = frame.copy()


# ----------- VIDEO STREAM ENDPOINT -----------
@app.route('/video')
def video_feed():
    def stream():
        global output_frame
        while True:
            with frame_lock:
                if output_frame is None:
                    continue
                ret, buffer = cv2.imencode(".jpg", output_frame)
                frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ----------- CONTROL ENDPOINTS -----------
@app.route('/start')
def start_detection():
    global running
    running = True
    return jsonify({"status": "started"})

@app.route('/stop')
def stop_detection():
    global running
    running = False
    return jsonify({"status": "stopped"})


# ----------- HOME PAGE -----------
@app.route('/')
def index():
    return render_template("index.html")


if __name__ == "__main__":
    t = threading.Thread(target=detect_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)

