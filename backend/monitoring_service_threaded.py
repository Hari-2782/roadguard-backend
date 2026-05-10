import cv2
import numpy as np
import sqlite3
import json
import os
import time
from datetime import datetime
import threading
import warnings

# RankWarning import compatible with numpy versions
try:
    from numpy.polynomial.polyutils import RankWarning
except Exception:
    try:
        from numpy import RankWarning  # older numpy
    except Exception:
        RankWarning = Warning  # fallback

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

from centerline import detect_lane_lines, draw_lane_lines
from sign_detector import YOLOSignDetector, SignState, JUNCTION_CLASSES
from pothole_detector import PotholeDetector

USE_DL_LANE_DETECTOR = False
try:
    from dl_lane_detector import DLLaneDetector
    if os.path.exists("models/lane_detector.pth"):
        dl_lane_detector = DLLaneDetector("models/lane_detector.pth")
        USE_DL_LANE_DETECTOR = True
    else:
        pass
except Exception:
    pass

warnings.simplefilter("ignore", RankWarning)

# =========================
# ---- Global config  -----
# =========================

EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)
DB_PATH = "violations.db"
YOLO_WEIGHTS = r"runs/detect/custom_domain_Model/weights/best.pt"
POTHOLE_MODEL_PATH = "pothole_model.pt"

DEFAULT_SPEED_LIMIT_KMH = 50
OVERSPEED_COOLDOWN_FRAMES = 30
OVERSPEED_TOLERANCE_KMH = 8
DEFAULT_LANE_DEVIATION_BAND_PX = 40
LANE_COOLDOWN_FRAMES = 300
FALLBACK_KMH_PER_PIXEL = 1.0
DEFAULT_KMH_PER_PIXEL = 1.0
DEFAULT_SPEED_OFFSET = 0.0
CALIB_FILE = "calibration.json"

FLOW_ROI_Y_START = 0.50
FLOW_ROI_Y_END = 0.90
FLOW_ROI_X_START = 0.20
FLOW_ROI_X_END = 0.80
FLOW_USE_MEDIAN = True
SPEED_SMOOTHING_ALPHA = 0.20
FLOW_MIN_THRESHOLD = 0.5

JUNCTION_MIN_CONF = 0.70
SPEED_SIGN_MIN_CONF = 0.25

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            type TEXT,
            value REAL,
            details TEXT,
            vehicle_id TEXT,
            image_path TEXT
        )
    """)
    try:
        cursor.execute("ALTER TABLE events_v2 ADD COLUMN image_path TEXT")
    except:
        pass
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id TEXT PRIMARY KEY,
            label TEXT,
            speed_limit_kmh INTEGER,
            lane_sensitivity_px INTEGER,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn, cursor

def log_event(cursor, conn, event_type, vehicle_id, value, details):
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO events_v2(ts, type, value, details, vehicle_id) VALUES (?, ?, ?, ?, ?)",
            (ts, event_type, value, details, vehicle_id),
        )
        conn.commit()
        print(f"[LOG] {ts} | {event_type} | value={value} | {details}")
    except Exception as e:
        print(f"[ERROR] Logging event: {e}")

def load_vehicle_config(conn, vehicle_id: str):
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT label, speed_limit_kmh, lane_sensitivity_px, notes FROM vehicles WHERE vehicle_id=?",
            (vehicle_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"label": None, "speed_limit_kmh": None, "lane_sensitivity_px": None, "notes": None}
        return dict(row)
    except Exception:
        return {"label": None, "speed_limit_kmh": None, "lane_sensitivity_px": None, "notes": None}

def load_calibration(calib_path=CALIB_FILE):
    try:
        with open(calib_path, "r") as f:
            data = json.load(f)
        kmh_per_pixel = float(data.get("kmh_per_pixel", DEFAULT_KMH_PER_PIXEL))
        speed_offset = float(data.get("offset", DEFAULT_SPEED_OFFSET))
        return kmh_per_pixel, speed_offset
    except Exception:
        return FALLBACK_KMH_PER_PIXEL, DEFAULT_SPEED_OFFSET

def get_roi_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    y1 = int(h * FLOW_ROI_Y_START)
    y2 = int(h * FLOW_ROI_Y_END)
    x1 = int(w * FLOW_ROI_X_START)
    x2 = int(w * FLOW_ROI_X_END)
    mask[y1:y2, x1:x2] = 255
    return mask

def check_road_visible(frame, h, w):
    try:
        roi = frame[int(h * FLOW_ROI_Y_START):int(h * FLOW_ROI_Y_END), int(w * FLOW_ROI_X_START):int(w * FLOW_ROI_X_END)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([0, 0, 40])
        upper_gray = np.array([180, 50, 220])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
        return ratio > 0.1
    except:
        return False

def compute_optical_flow_speed(old_gray, frame_gray, p0, lk_params, h, w):
    if p0 is None or len(p0) < 10:
        roi_mask = get_roi_mask(h, w)
        p0 = cv2.goodFeaturesToTrack(
            old_gray, maxCorners=300, qualityLevel=0.2, 
            minDistance=7, blockSize=7, mask=roi_mask
        )
        if p0 is None or len(p0) < 10:
            return 0.0, p0
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is None:
        return 0.0, None
    
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    if len(good_new) < 5:
        return 0.0, None
        
    deltas = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        dy = new[1] - old[1]
        if dy > 0: 
            deltas.append(dy)
            
    if not deltas:
        return 0.0, good_new.reshape(-1, 1, 2)
        
    deltas = np.array(deltas)
    if FLOW_USE_MEDIAN:
        flow_value = np.median(deltas)
    else:
        flow_value = np.mean(deltas)
        
    if flow_value < FLOW_MIN_THRESHOLD:
        flow_value = 0.0
        
    return flow_value, good_new.reshape(-1, 1, 2)

class RoadMonitorService:
    def __init__(self, video_source=0, vehicle_id="unknown"):
        self.video_source = video_source
        self.vehicle_id = vehicle_id
        
        self.current_speed = 0.0
        self.current_limit = DEFAULT_SPEED_LIMIT_KMH
        self.active_alerts = []
        
        self.frame_buffer = None
        self.buffer_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.capture_thread = None

        # Init DB and Models inside start() or init? 
        # Better in init but models are heavy. The user requested "Check api.py", so let's stick to init.
        
        self.engine = None
        if pyttsx3:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 150)
            except: pass

        self.conn, self.cursor = init_db()
        self.kmh_per_pixel, self.speed_offset = load_calibration()
        
        print("[INFO] Loading models...")
        self.sign_detector = YOLOSignDetector(YOLO_WEIGHTS)
        self.sign_state = SignState()
        self.pothole_detector = PotholeDetector(POTHOLE_MODEL_PATH)
        
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.start_capture()

    def start_capture(self):
        if self.capture_thread and self.capture_thread.is_alive():
            return
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop_capture(self):
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source {self.video_source}")
            return

        speed_buffer = []
        smoothed_speed = 0.0
        overspeed_cooldown = 0
        p0 = None
        old_gray = None
        
        while not self.stop_event.is_set():
            success, frame = cap.read()
            if not success:
                # auto-loop for file
                if isinstance(self.video_source, str) and os.path.exists(self.video_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            try:
                h, w = frame.shape[:2]
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if old_gray is None:
                    old_gray = frame_gray
                    continue

                # 1. Speed
                road_visible = check_road_visible(frame, h, w)
                raw_speed = 0.0
                if road_visible:
                    flow_value, p0 = compute_optical_flow_speed(old_gray, frame_gray, p0, self.lk_params, h, w)
                    old_gray = frame_gray.copy()
                    raw_speed = (flow_value * self.kmh_per_pixel) + self.speed_offset
                    if raw_speed < 0: raw_speed = 0.0
                else:
                    old_gray = frame_gray.copy()

                speed_buffer.append(raw_speed)
                if len(speed_buffer) > 10: speed_buffer.pop(0)
                buffer_median = np.median(speed_buffer) if speed_buffer else raw_speed
                smoothed_speed = SPEED_SMOOTHING_ALPHA * buffer_median + (1 - SPEED_SMOOTHING_ALPHA) * smoothed_speed
                self.current_speed = smoothed_speed

                # 2. Signs
                dets = self.sign_detector.detect(frame)
                is_valid_sign, sign_conf, sign_cls, sign_box = self.sign_detector.filter_signs(
                    dets, h, w, 
                    speed_min_conf=SPEED_SIGN_MIN_CONF,
                    junction_min_conf=JUNCTION_MIN_CONF
                )
                
                current_sign_limit = None
                if is_valid_sign:
                    self.sign_state.update(sign_cls, sign_conf)
                    current_sign_limit = self.sign_state.get_current_speed_limit()

                # 3. Limits
                # Note: loading config every frame is bad for perf, maybe cache or throttle
                cfg = load_vehicle_config(self.conn, self.vehicle_id)
                if current_sign_limit is not None:
                     self.current_limit = int(current_sign_limit)
                elif cfg.get("speed_limit_kmh") is not None:
                     self.current_limit = int(cfg.get("speed_limit_kmh"))
                else:
                     self.current_limit = DEFAULT_SPEED_LIMIT_KMH

                # 4. Alerts
                self.active_alerts = []
                if overspeed_cooldown > 0: overspeed_cooldown -= 1
                
                if self.current_speed > self.current_limit + OVERSPEED_TOLERANCE_KMH:
                    alert_msg = f"OVERSPEED: {self.current_speed:.1f}/{self.current_limit}"
                    self.active_alerts.append(alert_msg)
                    
                    if overspeed_cooldown == 0:
                        detail = f"{alert_msg} km/h"
                        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"overspeed_{self.vehicle_id}_{ts_file}.jpg"
                        filepath = os.path.join(EVIDENCE_DIR, filename)
                        
                        evidence_frame = frame.copy()
                        cv2.putText(evidence_frame, detail, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        cv2.imwrite(filepath, evidence_frame)
                        
                        log_event(self.cursor, self.conn, "overspeed", self.vehicle_id, float(self.current_speed), detail)
                        if self.engine:
                            try:
                                self.engine.say("Overspeed")
                                self.engine.runAndWait()
                            except: pass
                        overspeed_cooldown = OVERSPEED_COOLDOWN_FRAMES

                # 5. Draw
                disp = frame.copy()
                disp = YOLOSignDetector.draw(disp, dets)
                
                color = (0, 255, 0) if self.current_speed <= self.current_limit else (0, 0, 255)
                cv2.putText(disp, f"Speed: {self.current_speed:.1f} km/h", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(disp, f"Limit: {self.current_limit} km/h", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                y_alert = 100
                for alert in self.active_alerts:
                    cv2.putText(disp, alert, (20, y_alert), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_alert += 30

                ret, buffer = cv2.imencode('.jpg', disp)
                if ret:
                    with self.buffer_lock:
                        self.frame_buffer = buffer.tobytes()

            except Exception as e:
                print(f"[ERROR] processing loop: {e}")
                time.sleep(0.1)

        cap.release()

    def generate_frames(self):
        """Yields frames from the buffer"""
        while True:
            with self.buffer_lock:
                if self.frame_buffer:
                    frame = self.frame_buffer
                else:
                    frame = None
            
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04) # ~25 FPS
