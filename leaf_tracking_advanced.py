import cv2
import time
import RPi.GPIO as GPIO
from ultralytics import YOLO
from pan_tilt_control import set_angle, PAN_PIN, TILT_PIN

# ----------------------------
# PUMP RELAY CONFIG
# ----------------------------
PUMP_PIN = 26  # relay signal pin
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PUMP_PIN, GPIO.OUT)
GPIO.output(PUMP_PIN, GPIO.LOW)

# ----------------------------
# SERVO SETUP
# ----------------------------
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan = GPIO.PWM(PAN_PIN, 50)
tilt = GPIO.PWM(TILT_PIN, 50)
pan.start(7.5)
tilt.start(7.5)

pan_angle = 90
tilt_angle = 90

PAN_MIN, PAN_MAX = 40, 140
TILT_MIN, TILT_MAX = 50, 120

def clamp(v, mn, mx):
    return max(mn, min(mx, v))

def move_pan(angle):
    global pan_angle
    pan_angle = clamp(angle, PAN_MIN, PAN_MAX)
    set_angle(pan, pan_angle)

def move_tilt(angle):
    global tilt_angle
    tilt_angle = clamp(angle, TILT_MIN, TILT_MAX)
    set_angle(tilt, tilt_angle)

# ----------------------------
# MODEL SETUP
# ----------------------------
model_B = YOLO("best.pt")
model_C = YOLO("best2.pt")

DISEASE_KEYWORDS = ["blight", "spot", "mold", "mite", "rust"]

# ----------------------------
# AUTO SCAN CONFIG
# ----------------------------
SCAN_SPEED = 1.2
SCAN_DELAY = 0.03
scan_direction = 1
lost_timer = 0
tracking_mode = False

def auto_scan():
    global scan_direction, pan_angle
    pan_angle += scan_direction * SCAN_SPEED
    
    if pan_angle >= PAN_MAX:
        scan_direction = -1
    elif pan_angle <= PAN_MIN:
        scan_direction = 1
    
    move_pan(pan_angle)
    time.sleep(SCAN_DELAY)

# ----------------------------
# AUTO ZOOM (TILT)
# ----------------------------
def auto_zoom(box, frame_h):
    global tilt_angle
    x1,y1,x2,y2 = box
    height = y2 - y1

    # If leaf is too small → tilt down (zoom in)
    if height < frame_h * 0.25:
        tilt_angle += 1
        move_tilt(tilt_angle)

    # If leaf too close → tilt up
    elif height > frame_h * 0.45:
        tilt_angle -= 1
        move_tilt(tilt_angle)

# ----------------------------
# PUMP TRIGGER (SPRAY)
# ----------------------------
def spray_now():
    print("⚠️ Disease detected → ACTIVATING SPRAYER")
    GPIO.output(PUMP_PIN, GPIO.HIGH)
    time.sleep(1.2)   # spray duration
    GPIO.output(PUMP_PIN, GPIO.LOW)

# ----------------------------
# CHOOSE PRIORITY TARGET
# ----------------------------
def select_priority(boxes, labels):
    if not boxes:
        return None

    # 1. Priority: disease labels
    for box, lbl in zip(boxes, labels):
        l = lbl.lower()
        if any(d in l for d in DISEASE_KEYWORDS):
            return box, lbl, True  # (box, label, disease_flag)

    # 2. Else choose largest leaf
    areas = []
    for b in boxes:
        x1,y1,x2,y2 = b
        areas.append((x2-x1)*(y2-y1))
    i = areas.index(max(areas))
    return boxes[i], labels[i], False

# ----------------------------
# MAIN CAMERA LOOP
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    h,w,_ = frame.shape

    r1 = model_B(frame, verbose=False)
    r2 = model_C(frame, verbose=False)

    # extract boxes + labels from both models
    boxes, labels = [], []
    for r in (r1, r2):
        if len(r[0].boxes) > 0:
            for b,cidx in zip(r[0].boxes.xyxy.cpu(), r[0].boxes.cls.cpu()):
                box = b.numpy().astype(int)
                label = r[0].names[int(cidx)]
                boxes.append(box)
                labels.append(label)

    # ---------------------------
    # NO LEAF DETECTED → SCAN
    # ---------------------------
    if len(boxes) == 0:
        lost_timer += 1
        tracking_mode = False

        if lost_timer > 8:
            auto_scan()
        
        cv2.putText(frame, "SCANNING...", (10,30), 1,1,(255,255,0),2)
        cv2.imshow("Leaf Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # Reset lost timer
    lost_timer = 0
    tracking_mode = True

    # --------------------------------
    # SELECT PRIORITY TARGET
    # --------------------------------
    box, label, is_disease = select_priority(boxes, labels)
    x1,y1,x2,y2 = box
    cx, cy = (x1+x2)//2, (y1+y2)//2

    # Tracking movement
    dx = cx - w//2
    dy = cy - h//2

    pan_angle -= dx * 0.05
    tilt_angle += dy * 0.05

    move_pan(pan_angle)
    move_tilt(tilt_angle)

    # Auto zoom
    auto_zoom(box, h)

    # If disease detected → spray
    if is_disease:
        spray_now()

    color = (0,0,255) if is_disease else (0,255,0)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    cv2.putText(frame,label,(x1,y1-10),1,1,color,2)

    mode_text = "TRACKING (DISEASE!)" if is_disease else "TRACKING LEAF"
    cv2.putText(frame, mode_text, (10,30), 1,1,color,2)

    cv2.imshow("Leaf Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
