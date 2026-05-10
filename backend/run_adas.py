"""
ADAS Inference Engine - integrated with project structure.
Accepts --source argument (0 for webcam, or path to video file).
"""
import cv2
import torch
import numpy as np
import os
import sys
import threading
import time
import argparse
import sqlite3
from datetime import datetime
from collections import defaultdict, deque

# Project root is this file's parent directory (Final_Year_Research/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.utils.road_model import SimpleFastSCNN
from ultralytics import YOLO

try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

# ─── Config ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAZARD_MODEL_PATH = "models/hazard_model/best.pt"
ROAD_MODEL_PATH   = "models/road_segmentation/best.pth"
EVIDENCE_DIR      = "evidence"
DB_PATH           = "violations.db"
OUTPUT_DIR        = "outputs/videos"

ALPHA          = 0.7
GREEN_TH       = 0.20
ORANGE_TH      = 0.45
ROAD_OVERLAP_TH = 0.2
BOTTOM_GATE    = 0.7
PERSIST_FRAMES = 2
CLEAR_HOLD     = 12
SOUND_COOL     = 2.0

os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Model Loading ─────────────────────────────────────────
print("[INFO] Loading hazard detector (YOLOv8)...")
hazard_model = YOLO(HAZARD_MODEL_PATH)

print("[INFO] Loading road segmenter (SimpleFastSCNN)...")
road_model = SimpleFastSCNN().to(DEVICE)
try:
    ckpt = torch.load(ROAD_MODEL_PATH, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    road_model.load_state_dict(state, strict=False)
    print("[INFO] Road model loaded.")
except Exception as e:
    print(f"[WARN] Road model load failed ({e}), using fallback heuristic.")
    road_model = None
road_model_loaded = road_model
if road_model:
    road_model.eval()

# ─── Helpers ───────────────────────────────────────────────
last_sound_time = 0.0

def play_sound(freq, duration=200):
    global last_sound_time
    now = time.time()
    if HAS_SOUND and now - last_sound_time > SOUND_COOL:
        threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()
        last_sound_time = now

def preprocess(frame):
    img = cv2.resize(frame, (384, 384))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

def get_road_mask(frame):
    if road_model is None:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.4):, :] = 255
        return mask
    with torch.no_grad():
        pred = road_model(preprocess(frame))[0, 0]
    mask = (pred > 0.5).cpu().numpy().astype(np.uint8)
    return cv2.resize(mask, (frame.shape[1], frame.shape[0]))

def log_event(etype, value, frame):
    ts = datetime.now()
    filename = f"{etype}_{ts.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    cv2.imwrite(os.path.join(EVIDENCE_DIR, filename), frame)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO events_v2 (ts, type, value, details, vehicle_id, image_path, risk_level) VALUES (?,?,?,?,?,?,?)",
            (ts.isoformat(), etype, value, etype, "DEMO-CAR-01", filename, "HIGH")
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] {e}")

# ─── Main ──────────────────────────────────────────────────
def run(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_src if 1 < fps_src < 120 else 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Save output video
    is_file = isinstance(source, str) and os.path.isfile(source)
    out_path = None
    writer = None
    if is_file:
        out_name = os.path.splitext(os.path.basename(source))[0] + "_adas.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"[INFO] Output will be saved to: {out_path}")

    track_history = defaultdict(lambda: deque(maxlen=PERSIST_FRAMES))
    prev_risk = 0.0
    clear_counter = 0
    frame_count = 0
    road_mask_cached = None
    last_log_risk = 0.0

    print(f"[INFO] Running ADAS on source: {source}. Press ESC to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Road mask (update every 2 frames)
        if frame_count % 2 == 0 or road_mask_cached is None:
            road_mask_cached = get_road_mask(frame)
        road_mask = road_mask_cached

        results = hazard_model(frame, conf=0.35, verbose=False)[0]
        risk_now = 0.0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi = road_mask[y1:y2, x1:x2]
            overlap = roi.sum() / max(1, (x2 - x1) * (y2 - y1))
            if overlap < ROAD_OVERLAP_TH:
                continue

            raw_bot = y2 / h
            pos = max(0.0, min(1.0, (raw_bot - BOTTOM_GATE) / (1.0 - BOTTOM_GATE)))
            risk_now = max(risk_now, pos)

            key = (x1 // 80, y1 // 80)
            track_history[key].append(pos)
            if len(track_history[key]) < PERSIST_FRAMES:
                continue

            if pos >= ORANGE_TH:
                col, lbl = (0, 0, 255), "HIGH RISK (Immediate Hazard)"
            elif pos >= GREEN_TH:
                col, lbl = (0, 165, 255), "MEDIUM RISK (Approaching)"
            else:
                col, lbl = (0, 255, 0), "LOW RISK (Object Ahead)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, lbl, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        risk = ALPHA * prev_risk + (1 - ALPHA) * risk_now
        prev_risk = risk

        if risk >= ORANGE_TH:
            clear_counter = 0
            cv2.putText(frame, "HIGH RISK - BRAKE!", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            play_sound(1200, 300)
            if risk > last_log_risk + 0.1:
                log_event("HIGH_RISK_HAZARD", risk, frame)
                last_log_risk = risk
        elif risk >= GREEN_TH:
            clear_counter = 0
            cv2.putText(frame, "MEDIUM RISK - CAUTION", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            play_sound(800, 150)
        else:
            clear_counter += 1
            last_log_risk = max(0, last_log_risk - 0.05)
            if 0 < clear_counter < CLEAR_HOLD:
                cv2.putText(frame, "HAZARD CLEARED", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Risk bar
        bar_w = 200
        bar_fill = int(risk * bar_w)
        bar_col = (0, 255, 0) if risk < GREEN_TH else ((0, 165, 255) if risk < ORANGE_TH else (0, 0, 255))
        cv2.rectangle(frame, (40, 75), (40 + bar_w, 95), (60, 60, 60), -1)
        if bar_fill > 0:
            cv2.rectangle(frame, (40, 75), (40 + bar_fill, 95), bar_col, -1)
        cv2.putText(frame, f"Risk: {risk:.2f}", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, "AI ROAD SAFETY SYSTEM | Road-Filtered ADAS",
                    (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if writer:
            writer.write(frame)

        cv2.imshow("ADAS - AI Road Safety (press ESC to stop)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if out_path:
        print(f"[DONE] Output saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="Video source: 0/1 for webcam, or path to a video file")
    args = parser.parse_args()

    src = args.source
    if src.isdigit():
        src = int(src)

    run(src)
