import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

# Servo pins
PAN_PIN = 17
TILT_PIN = 27

GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

# PWM: 50 Hz for SG90/MG90S servos
pan = GPIO.PWM(PAN_PIN, 50)
tilt = GPIO.PWM(TILT_PIN, 50)

pan.start(7.5)   # Center position
tilt.start(7.5)  # Center position

def set_angle(pwm, angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.05)

def center_camera():
    set_angle(pan, 90)
    set_angle(tilt, 90)

def move_left():
    set_angle(pan, 120)

def move_right():
    set_angle(pan, 60)

def move_up():
    set_angle(tilt, 60)

def move_down():
    set_angle(tilt, 120)

def cleanup():
    pan.stop()
    tilt.stop()
    GPIO.cleanup()

if __name__ == "__main__":
    try:
        center_camera()
        time.sleep(1)
        move_left()
        time.sleep(1)
        move_right()
        time.sleep(1)
        move_up()
        time.sleep(1)
        move_down()
        time.sleep(1)
        center_camera()
    except KeyboardInterrupt:
        cleanup()
import pigpio
import time

# -------------------------
# PIN SETTINGS (BCM)
# -------------------------
PAN_PIN = 17      # Servo 1
TILT_PIN = 27     # Servo 2

# -------------------------
# SERVO LIMITS
# -------------------------
PAN_MIN = 500
PAN_MAX = 2500

TILT_MIN = 500
TILT_MAX = 2500

# Start centered
pan_pos = 1500
tilt_pos = 1500

# -------------------------
# Initialize pigpio
# -------------------------
pi = pigpio.pi()
if not pi.connected:
    print("ERROR: pigpio daemon not running! Run: sudo systemctl start pigpiod")
    exit()

pi.set_servo_pulsewidth(PAN_PIN, pan_pos)
pi.set_servo_pulsewidth(TILT_PIN, tilt_pos)

# -------------------------
# MOVE FUNCTIONS
# -------------------------
def move_pan(delta):
    global pan_pos
    pan_pos = max(PAN_MIN, min(PAN_MAX, pan_pos + delta))
    pi.set_servo_pulsewidth(PAN_PIN, pan_pos)

def move_tilt(delta):
    global tilt_pos
    tilt_pos = max(TILT_MIN, min(TILT_MAX, tilt_pos + delta))
    pi.set_servo_pulsewidth(TILT_PIN, tilt_pos)

def center_camera():
    global pan_pos, tilt_pos
    pan_pos = 1500
    tilt_pos = 1500
    pi.set_servo_pulsewidth(PAN_PIN, pan_pos)
    pi.set_servo_pulsewidth(TILT_PIN, tilt_pos)
    time.sleep(0.3)

def shutdown():
    pi.set_servo_pulsewidth(PAN_PIN, 0)
    pi.set_servo_pulsewidth(TILT_PIN, 0)
    pi.stop()
