import cv2
import time
from ultralytics import YOLO
import RPi.GPIO as GPIO
from pan_tilt_control import set_angle, PAN_PIN, TILT_PIN

# Initialize GPIO (sharing from pan_tilt_control)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan = GPIO.PWM(PAN_PIN, 50)
tilt = GPIO.PWM(TILT_PIN, 50)
pan.start(7.5)
tilt.start(7.5)

# Starting servo angles
pan_angle = 90
tilt_angle = 90

# Load models
model_B = YOLO("best.pt")
model_C = YOLO("best2.pt")

def clamp(x, mn, mx):
    return max(mn, min(mx, x))

def adjust_servos(cx, cy, frame_w, frame_h):
    global pan_angle, tilt_angle

    center_x = frame_w // 2
    center_y = frame_h // 2

    dx = cx - center_x
    dy = cy - center_y

    # Sensitivity
    kx = 0.05
    ky = 0.05

    # Update angles
    pan_angle -= dx * kx
    tilt_angle += dy * ky

    # Clamp angles
    pan_angle = clamp(pan_angle, 40, 140)
    tilt_angle = clamp(tilt_angle, 40, 140)

    # Move servos
    set_angle(pan, pan_angle)
    set_angle(tilt, tilt_angle)

def get_largest_box(results):
    boxes = results[0].boxes
    if not boxes:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
    idx = areas.argmax()

    return xyxy[idx]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape

    # Run both models
    r1 = model_B(frame, verbose=False)
    r2 = model_C(frame, verbose=False)

    # Pick best detection (largest leaf)
    box1 = get_largest_box(r1)
    box2 = get_largest_box(r2)

    box = None
    if box1 is not None and box2 is not None:
        box = box1 if (box1[2]-box1[0]) * (box1[3]-box1[1]) > (box2[2]-box2[0]) * (box2[3]-box2[1]) else box2
    else:
        box = box1 if box1 is not None else box2

    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        cx = (x1 + x2)//2
        cy = (y1 + y2)//2

        # Draw bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

        # Adjust camera towards the leaf
        adjust_servos(cx, cy, frame_w, frame_h)

    cv2.imshow("Leaf Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
import cv2
import time
from ultralytics import YOLO
import RPi.GPIO as GPIO
from pan_tilt_control import set_angle, PAN_PIN, TILT_PIN

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan = GPIO.PWM(PAN_PIN, 50)
tilt = GPIO.PWM(TILT_PIN, 50)
pan.start(7.5)
tilt.start(7.5)

# Servo angles
pan_angle = 90
tilt_angle = 90

# Angle limits
PAN_MIN = 40
PAN_MAX = 140

# Load YOLO models
model_B = YOLO("best.pt")
model_C = YOLO("best2.pt")

def clamp(v, mn, mx):
    return max(mn, min(mx, v))

def move_pan(angle):
    global pan_angle
    pan_angle = clamp(angle, PAN_MIN, PAN_MAX)
    set_angle(pan, pan_angle)

def move_tilt(angle):
    global tilt_angle
    tilt_angle = clamp(angle, 40, 140)
    set_angle(tilt, tilt_angle)

def get_largest_box(results):
    boxes = results[0].boxes
    if not boxes:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
    idx = areas.argmax()
    return xyxy[idx]

def adjust_servos(cx, cy, w, h):
    global pan_angle, tilt_angle

    dx = cx - w//2
    dy = cy - h//2

    pan_angle -= dx * 0.05
    tilt_angle += dy * 0.05

    move_pan(pan_angle)
    move_tilt(tilt_angle)

# -------------------------------
# MODE B: SCANNING LEFT TO RIGHT
# -------------------------------

SCAN_SPEED = 1.2     # degrees per update
SCAN_DELAY = 0.03    # delay between movements

scan_direction = 1    # 1 = right, -1 = left

def auto_scan():
    """Moves camera slowly left-right-left looking for leaves."""
    global pan_angle, scan_direction

    pan_angle += SCAN_SPEED * scan_direction

    if pan_angle >= PAN_MAX:
        scan_direction = -1
    elif pan_angle <= PAN_MIN:
        scan_direction = 1

    move_pan(pan_angle)
    time.sleep(SCAN_DELAY)

# --------------------------
# CAMERA LOOP
# --------------------------

cap = cv2.VideoCapture(0)

# Mode flags
tracking_mode = False
no_leaf_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Run YOLO model B & C
    r1 = model_B(frame, verbose=False)
    r2 = model_C(frame, verbose=False)

    box1 = get_largest_box(r1)
    box2 = get_largest_box(r2)

    # Select best detection
    box = None
    if box1 is not None and box2 is not None:
        a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        box = box1 if a1 > a2 else box2
    else:
        box = box1 if box1 is not None else box2

    # -----------------------------------
    # MODE SWITCHING: SCAN vs TRACK
    # -----------------------------------

    if box is None:
        tracking_mode = False
        no_leaf_frames += 1

        # If no leaf for more than 5 frames → start auto scan
        if no_leaf_frames > 5:
            auto_scan()

    else:
        tracking_mode = True
        no_leaf_frames = 0

        x1, y1, x2, y2 = box.astype(int)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Draw box and center point
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

        # Now track leaf by moving servos
        adjust_servos(cx, cy, w, h)

    # Status text overlay
    status = "TRACKING MODE" if tracking_mode else "SCANNING..."
    cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Leaf Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
import cv2
import time
import numpy as np
from ultralytics import YOLO
from pan_tilt_control import move_pan, move_tilt, center_camera
from config import (
    OD_MODEL_PATH_B, OD_MODEL_PATH_C,
    OD_TO_CLF_MAP_B, OD_TO_CLF_MAP_C,
    DEBUG_OD_CONF_THRESHOLD, OD_IOU_THRESHOLD
)

# -------------------------
# LOAD YOLO MODELS
# -------------------------
print("Loading YOLO models...")
model_B = YOLO(OD_MODEL_PATH_B)
model_C = YOLO(OD_MODEL_PATH_C)

names_B = list(model_B.names.values())
names_C = list(model_C.names.values())
print("Models loaded.")


# -------------------------
# CAMERA
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not detected.")
    exit()

FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

TRACK_BOX_COLOR = (0, 255, 255)

# -------------------------
# AUTO-SCAN VARIABLES
# -------------------------
scan_direction = 1
scan_speed = 15
last_detection_time = time.time()


# -------------------------
# DETECTION FUNCTION
# -------------------------
def detect(model, frame, thresh, model_names):
    labels, confs, boxes = [], [], []
    res = model(frame, conf=thresh, iou=OD_IOU_THRESHOLD, imgsz=416, verbose=False, device="cpu")

    if len(res) > 0 and len(res[0].boxes) > 0:
        b = res[0].boxes
        xyxy = b.xyxy.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)

        for box, c, k in zip(xyxy, conf, cls):
            if 0 <= k < len(model_names):
                labels.append(model_names[k])
                confs.append(c)
                boxes.append(tuple(box))

    return labels, confs, boxes


# -------------------------
# TRACKING LOOP
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect from BOTH models
    labels_B, conf_B, boxes_B = detect(model_B, frame, DEBUG_OD_CONF_THRESHOLD, names_B)
    labels_C, conf_C, boxes_C = detect(model_C, frame, DEBUG_OD_CONF_THRESHOLD, names_C)

    all_boxes = boxes_B + boxes_C

    if len(all_boxes) > 0:
        last_detection_time = time.time()

        # Pick the BIGGEST box
        biggest = max(all_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = biggest

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), TRACK_BOX_COLOR, 3)

        # Find center of leaf
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # -------------------------
        # MOVE PAN/TILT TOWARD LEAF
        # -------------------------
        dx = cx - CENTER_X
        dy = cy - CENTER_Y

        if abs(dx) > 30:
            move_pan(-dx // 20)

        if abs(dy) > 30:
            move_tilt(dy // 20)

    else:
        # -------------------------
        # AUTO-SCAN MODE (NO DETECTION)
        # -------------------------
        if time.time() - last_detection_time > 1.5:
            move_pan(scan_speed * scan_direction)
            time.sleep(0.05)

            # Reverse on limits
            if np.random.rand() < 0.02:
                scan_direction *= -1

    # Show (if using X11/VNC)
    cv2.imshow("Leaf Tracking (pigpio)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
