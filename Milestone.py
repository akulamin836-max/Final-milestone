import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# ---------------- GPIO SETUP ----------------
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Pin Definitions
IN1, IN2, IN3, IN4 = 21, 20, 16, 12
GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)

# ---------------- PWM SETUP ----------------
# Frequency = 100Hz (Good for standard DC motors)
LEFT_PWM = GPIO.PWM(IN1, 255)
RIGHT_PWM = GPIO.PWM(IN3, 255)
LEFT_PWM.start(0)
RIGHT_PWM.start(0)

# ---------------- CONTROL SETTINGS ----------------
BASE_SPEED = 30       # Forward speed
KP = 0.60            # Turning sensitivity
KD = 0.1              # Smoothness
MAX_CORRECTION = 70   # Prevents reversing (Keep <= BASE_SPEED)
last_error = 0

# ---------------- MOTOR FUNCTIONS ----------------
def drive(left_speed, right_speed):
    # Lock direction to FORWARD only
    GPIO.output(IN2, 0) 
    GPIO.output(IN4, 0)

    # Ensure speeds stay between 0 and 100
    l_duty = max(0, min(int(left_speed), 255))
    r_duty = max(0, min(int(right_speed), 255))

    LEFT_PWM.ChangeDutyCycle(l_duty)
    RIGHT_PWM.ChangeDutyCycle(r_duty)

def stop():
    LEFT_PWM.ChangeDutyCycle(0)
    RIGHT_PWM.ChangeDutyCycle(0)

# ---------------- CAMERA SETUP ----------------
# Use V4L2 to prevent Segmentation Faults
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
time.sleep(2)

print("Robot Active: Red Centroid Tracking")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # 1. Processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Red Mask (Lower and Upper Red)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # 3. Clean Noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 4. Find Centroid
        M = cv2.moments(mask)
        
        if M["m00"] > 500: # Threshold
            cx = int(M["m10"] / M["m00"])
            
            # Error Calculation (Center is 160)
            error = cx - 160 
            
            # PD Control
            derivative = error - last_error
            correction = (KP * error) + (KD * derivative)
            correction = max(-MAX_CORRECTION, min(MAX_CORRECTION, correction))

            # --- THE FIX: SWAPPED SIGNS ---
            # Scenario: Red is on LEFT (error is negative, e.g., -50)
            # We need RIGHT motor > LEFT motor to turn Left.
            
            # left_speed = 30 + (-value) = SLOWER
            # right_speed = 30 - (-value) = FASTER
            left_speed = BASE_SPEED - correction
            right_speed = BASE_SPEED + correction

            drive(left_speed, right_speed)
            last_error = error

            # Visual Feedback
            cv2.circle(frame, (cx, 120), 10, (0, 255, 0), -1)
        else:
            stop()

        # Debug Display
        cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), 2)
        cv2.imshow("Red Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()