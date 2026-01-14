import os
import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import torch
import einops
import torch.nn as nn
import torchvision.models as models
from clock_utils import warp

# ================== CONFIG ==================
YOLO_MODEL_PATH = r"D:\Bai tap\Visual Studio for Python\Midterm\YOLO_customized.pt"
CLOCK_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.7
PROCESS_INTERVAL = 2.0   # seconds
CROP_MARGIN = 0.12

TIME_MODEL_PATH = r"D:\Bai tap\Visual Studio for Python\Midterm\full.pth"
STN_MODEL_PATH  = r"D:\Bai tap\Visual Studio for Python\Midterm\full_st.pth"

# ================== LOAD YOLO ==================
yolo = YOLO(YOLO_MODEL_PATH)

# ================== LOAD CLOCK READER ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_stn = models.resnet50(pretrained=False)
model_stn.fc = nn.Linear(2048, 8 )

model_time = models.resnet50(pretrained=False)
model_time.fc = nn.Linear(2048, 720)

model_time.load_state_dict(torch.load(TIME_MODEL_PATH, map_location=device))
model_stn.load_state_dict(torch.load(STN_MODEL_PATH, map_location=device))

model_time.to(device).eval()
model_stn.to(device).eval()

# ================== CLOCK READER FUNCTION ==================
def read_clock_time(cropped_img):
    img = cv2.resize(cropped_img, (224, 224)) / 255.0
    img = einops.rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_st = model_stn(img)
        pred_st = torch.cat([pred_st, torch.ones(1, 1, device=device)], dim=1)
        Minv_pred = pred_st.view(-1, 3, 3)

        img_warped = warp(img, Minv_pred)

        pred = model_time(img_warped)
        idx = torch.argmax(pred, dim=1)[0]

        hour = (idx // 60).item()
        minute = (idx % 60).item()

    return hour, minute

# ================== MAIN ==================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

cv2.namedWindow("Auto Clock Reader System", cv2.WINDOW_NORMAL)
# This allows you to drag the corner of the window to make it larger

last_process_time = 0.0
last_read_time = None

print("=== AUTO ANALOG CLOCK READER ===")


clock_results = {} # Stores {track_id: "HH:MM"}

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Use track() instead of predict() to get persistent IDs
    # persist=True keeps the IDs alive across frames
    results = yolo.track(frame, persist=True, verbose=False, conf=CONFIDENCE_THRESHOLD)

    current_time = time.time()
    can_process = (current_time - last_process_time) > PROCESS_INTERVAL

    for result in results:
        if result.boxes is None or result.boxes.id is None:
            continue

        # Get boxes, class IDs, and Track IDs
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        clss = result.boxes.cls.cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            if cls != CLOCK_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box)

            # ---- Draw bbox and ID ----
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ---- Read time (interval-based) ----
            if can_process:
                h, w = frame.shape[:2]
                bw, bh = x2 - x1, y2 - y1
                
                x1m = max(0, int(x1 - CROP_MARGIN * bw))
                y1m = max(0, int(y1 - CROP_MARGIN * bh))
                x2m = min(w, int(x2 + CROP_MARGIN * bw))
                y2m = min(h, int(y2 + CROP_MARGIN * bh))

                cropped_clock = frame[y1m:y2m, x1m:x2m].copy()

                if cropped_clock.size > 0:
                    hour, minute = read_clock_time(cropped_clock)
                    # Map the time to the unique Track ID
                    clock_results[track_id] = f"{hour:02d}:{minute:02d}"

            # ---- Display time for this specific ID ----
            if track_id in clock_results:
                time_display = clock_results[track_id]
                cv2.putText(frame, time_display, (x1, y1 - 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if can_process:
        last_process_time = current_time

    # ---- Overlay time ----
    if last_read_time is not None:
        cv2.putText(frame,
                    f"Time: {last_read_time}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2)

    if not can_process:
        cv2.putText(frame,
                    "Cooldown...",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

    cv2.imshow("Auto Clock Reader System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()