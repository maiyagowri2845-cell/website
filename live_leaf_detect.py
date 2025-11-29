import cv2
from detector import (
    od_model_B,
    od_model_C,
    OD_CLASS_NAMES_B_DYNAMIC,
    OD_CLASS_NAMES_C_DYNAMIC,
    run_od_prediction,
    ensemble_hard_voting,
    DISEASE_REMEDIES,
    ENSEMBLE_VOTING_THRESHOLD,
    DEBUG_OD_CONF_THRESHOLD,
    OD_TO_CLF_MAP_B,
    OD_TO_CLF_MAP_C
)
import time
import numpy as np


WINDOW_NAME = "Real-Time Leaf Disease Detection (Ensemble YOLO B + C)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Could not open the camera.")
    exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Failed to read camera frame.")
        break

    frame_count += 1

    # ----- Run Object Detection (Model B and C) -----
    labels_B, conf_B, boxes_B = run_od_prediction(
        od_model_B, frame, DEBUG_OD_CONF_THRESHOLD, OD_CLASS_NAMES_B_DYNAMIC
    )

    labels_C, conf_C, boxes_C = run_od_prediction(
        od_model_C, frame, DEBUG_OD_CONF_THRESHOLD, OD_CLASS_NAMES_C_DYNAMIC
    )

    # ----- Ensemble Input List -----
    od_predictions_list = [
        (labels_B, conf_B, OD_TO_CLF_MAP_B),
        (labels_C, conf_C, OD_TO_CLF_MAP_C),
    ]

    # ----- Ensemble Fusion -----
    ensembled_label, ensembled_conf = ensemble_hard_voting(od_predictions_list)

    # ----- Determine remedy/status -----
    remedy_key = "UNCERTAIN_MONITOR"

    if "UNCERTAIN" not in ensembled_label and ensembled_conf >= ENSEMBLE_VOTING_THRESHOLD:
        remedy_key = ensembled_label

    elif len(labels_B) == 0 and len(labels_C) == 0:
        ensembled_label = "NO LEAF DETECTED"
        remedy_key = "Tomato_Healthy"

    remedy_data = DISEASE_REMEDIES.get(remedy_key, DISEASE_REMEDIES["UNCERTAIN_MONITOR"])
    status_text = remedy_data["status"]
    remedy_text = remedy_data["remedies"]

    # ----- Draw Bounding Boxes -----
    for i, (x1, y1, x2, y2) in enumerate(boxes_B):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{labels_B[i]} {conf_B[i]:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    for i, (x1, y1, x2, y2) in enumerate(boxes_C):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
        cv2.putText(frame, f"{labels_C[i]} {conf_C[i]:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    # ----- FPS Calculation -----
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    header = f"FPS: {fps:.2f} | {ensembled_label} | Conf: {ensembled_conf:.2f}"
    cv2.putText(frame, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,255), 2)

    # ----- Remedy text -----
    y0 = frame.shape[0] - 100
    for line in remedy_text.split("\n"):
        cv2.putText(frame, line, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        y0 += 25

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
