from flask import Flask, Response
import cv2
import numpy as np
import time
from ultralytics import YOLO
from ensemble import ensemble_hard_voting
from config import (
    CLF_CLASS_NAMES, ENSEMBLE_VOTING_THRESHOLD, OD_IOU_THRESHOLD, 
    OD_MODEL_PATH_B, OD_MODEL_PATH_C, DISEASE_REMEDIES,
    OD_TO_CLF_MAP_B, OD_TO_CLF_MAP_C, DEBUG_OD_CONF_THRESHOLD
)

app = Flask(__name__)

od_model_B = YOLO(OD_MODEL_PATH_B)
od_model_C = YOLO(OD_MODEL_PATH_C)

OD_CLASS_NAMES_B = list(od_model_B.names.values())
OD_CLASS_NAMES_C = list(od_model_C.names.values())

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # YOLO inference
        labels_B, conf_B, boxes_B = run_od(od_model_B, frame)
        labels_C, conf_C, boxes_C = run_od(od_model_C, frame)

        od_list = [
            (labels_B, conf_B, OD_TO_CLF_MAP_B),
            (labels_C, conf_C, OD_TO_CLF_MAP_C)
        ]

        final_label, final_conf = ensemble_hard_voting(od_list)

        # Draw boxes
        for label, conf, box in zip(labels_B + labels_C, conf_B + conf_C, boxes_B + boxes_C):
            x1,y1,x2,y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run_od(model, frame):
    res = model(frame, conf=0.25, iou=OD_IOU_THRESHOLD, imgsz=416, verbose=False, device="cpu")
    labels, confs, boxes = [], [], []

    if len(res) and len(res[0].boxes):
        box = res[0].boxes
        xyxy = box.xyxy.cpu().numpy().astype(int)
        conf = box.conf.cpu().numpy()
        cls = box.cls.cpu().numpy().astype(int)
        names = list(model.names.values())

        for b, c, k in zip(xyxy, conf, cls):
            labels.append(names[k])
            confs.append(float(c))
            boxes.append(tuple(b))

    return labels, confs, boxes

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    print("Starting video stream server...")
    app.run(host="0.0.0.0", port=5000, debug=False)
