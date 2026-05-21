# Artificial Intelligence-Based Missile Guidance System
# YOLO + Raspberry Pi + Servo Control

#!/usr/bin/env python3
"""
yolo_servo_relay.py
- Uses Ultralytics YOLO (pretrained on COCO) to detect only 'car' and 'airplane'.
- Controls two servos (pan, tilt) to aim at detected object's center.
- Controls a relay (treated as LED on/off) when detection present.
- Optimized for Raspberry Pi 3B+: uses yolov8n, imgsz=320, and simple smoothing.
- Safe: relay used only for LED or benign loads. No ignition functionality.
"""
import time
import math
import signal
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from gpiozero import Servo, LED
from time import sleep

# -----------------------
# CONFIGURATION
# -----------------------
CAM_SRC = 0
MODEL_NAME = "yolov8n.pt"
IMG_SIZE = 320
CONF_THRESH = 0.25
PROCESS_EVERY_N_FRAMES = 1
DETECTION_CLASS_NAMES = ("car", "airplane")

PAN_PIN = 17
TILT_PIN = 27
RELAY_PIN = 22

SERVO_STEP = 0.08
DEADZONE_PIXELS = 6
MAX_FRAME_SKIP = 2

GAIN_X = 0.0025
GAIN_Y = 0.0025

# -----------------------
# HARDWARE SETUP CHECK
# -----------------------
print("Loading model and initializing hardware...")

model = YOLO(MODEL_NAME)

names_map = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else model.names

if isinstance(names_map, list):
    names_dict = {i: n for i, n in enumerate(names_map)}
else:
    names_dict = names_map

target_class_indices = [i for i, n in names_dict.items() if n in DETECTION_CLASS_NAMES]

if not target_class_indices:
    print("ERROR: Could not find requested classes in model names. Model classes include:")
    print(names_dict)
    sys.exit(1)

print(f"Target class indices for {DETECTION_CLASS_NAMES}: {target_class_indices}")

cap = cv2.VideoCapture(CAM_SRC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

ret, test_frame = cap.read()
if not ret:
    print("ERROR: cannot open camera. Check CAM_SRC and camera connection.")
    cap.release()
    sys.exit(1)

frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_center = (frame_w // 2, frame_h // 2)

print(f"Camera opened: resolution {frame_w}x{frame_h}, center {frame_center}")

servo_pan = Servo(PAN_PIN)
servo_tilt = Servo(TILT_PIN)
relay = LED(RELAY_PIN)

pan_pos = 0.0
tilt_pos = 0.0

try:
    servo_pan.value = pan_pos
    servo_tilt.value = tilt_pos
except Exception as e:
    print("Warning: could not set initial servo values. Check wiring and permissions.", e)

running = True

def signal_handler(sig, frame):
    global running
    print("\nReceived exit signal, cleaning up...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

frame_count = 0
processed_frames = 0
last_time = time.time()

print("Starting main loop. Press Ctrl+C to stop.")

while running:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        continue

    frame_count += 1
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        continue

    processed_frames += 1

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        verbose=False
    )

    if len(results) == 0 or len(results[0].boxes) == 0:
        relay.off()
        print(f"[{processed_frames}] No detections")

    else:
        boxes = results[0].boxes

        try:
            cls_array = boxes.cls.cpu().numpy()
            conf_array = boxes.conf.cpu().numpy()
            xyxy_array = boxes.xyxy.cpu().numpy()
        except Exception:
            cls_array = np.array(boxes.cls)
            conf_array = np.array(boxes.conf)
            xyxy_array = np.array(boxes.xyxy)

        mask = np.isin(cls_array.astype(int), target_class_indices)

        if np.any(mask):
            idxs = np.where(mask)[0]
            best_idx = idxs[np.argmax(conf_array[idxs])]

            x1, y1, x2, y2 = xyxy_array[best_idx].astype(int)
            conf = float(conf_array[best_idx])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            dx = cx - frame_center[0]
            dy = cy - frame_center[1]

            pixel_error = int(math.hypot(dx, dy))

            relay.on()

            if abs(dx) <= DEADZONE_PIXELS:
                dx = 0
            if abs(dy) <= DEADZONE_PIXELS:
                dy = 0

            delta_pan = -dx * GAIN_X
            delta_tilt = -dy * GAIN_Y

            pan_pos += delta_pan
            tilt_pos += delta_tilt

            pan_pos = max(-1.0, min(1.0, pan_pos))
            tilt_pos = max(-1.0, min(1.0, tilt_pos))

            try:
                servo_pan.value = float(pan_pos)
                servo_tilt.value = float(tilt_pos)
            except Exception as e:
                print("Servo write error:", e)

            print(
                f"[{processed_frames}] Detected target cls={int(cls_array[best_idx])} "
                f"conf={conf:.2f} bbox=({x1},{y1},{x2},{y2}) "
                f"center=({cx},{cy}) error={pixel_error}px "
                f"pan={pan_pos:.3f} tilt={tilt_pos:.3f}"
            )

        else:
            relay.off()
            print(f"[{processed_frames}] Detections present but none are target classes.")

    now = time.time()
    elapsed = now - last_time
    if elapsed >= 1.0:
        fps = processed_frames / elapsed
        print(f"Processing FPS ~ {fps:.2f}")
        processed_frames = 0
        last_time = now

print("Cleaning up...")
cap.release()
servo_pan.detach()
servo_tilt.detach()
relay.off()
print("Done. Exiting.")

