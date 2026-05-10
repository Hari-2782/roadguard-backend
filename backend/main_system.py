import cv2
import time
import logging
import sqlite3
import os
import threading
import queue
from datetime import datetime
import numpy as np
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# Modules
from modules.perception.road_segmenter import RoadSegmenter
from modules.perception.hazard_detector import HazardDetector
from modules.perception.pothole_detector import PotholeDetector
from modules.perception.sign_detector import SignDetector
from modules.perception.lane_detector import LaneDetector
from modules.behavior.imu_behavior import IMUBehavior
from modules.fusion.risk_engine import RiskEngine
from modules.mapping.pothole_map_logger import PotholeMapLogger
from modules.utils.gps_reader import GPSReader
from modules.utils.event_types import EventType
from modules.utils.kalman_filter import RiskKalmanFilter

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MainSystem")

# Constants
DB_PATH = "violations.db" 
SAFETY_DB_PATH = "database/safety_events.db"
EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

class AudioAlerter:
    def __init__(self):
        self.queue = queue.Queue()
        self.engine = None
        if pyttsx3:
            try:
                self.engine = pyttsx3.init()
            except:
                logger.warning("Failed to init TTS engine")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.last_alert_time = 0

    def _worker(self):
        while True:
            msg = self.queue.get()
            if self.engine:
                try:
                    self.engine.say(msg)
                    self.engine.runAndWait()
                except Exception as e:
                    logger.error(f"TTS Error: {e}")
            self.queue.task_done()

    def alert(self, message, cooldown=5):
        now = time.time()
        if now - self.last_alert_time > cooldown:
            self.queue.put(message)
            self.last_alert_time = now

class RoadSafetySystem:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            test_video = "Videos_test/WhiteLineCrossing.mp4"
            if os.path.exists(test_video):
                logger.warning(f"Using test video: {test_video}")
                self.cap = cv2.VideoCapture(test_video)
            else:
                logger.error("No video source found.")
                # Don't raise, just let it fail gracefully in run loop
        
        # Initialize Modules
        logger.info("Initializing AI Modules...")
        self.road_segmenter = RoadSegmenter(model_path="models/road_segmentation/best.pth")
        self.hazard_detector = HazardDetector()
        self.pothole_detector = PotholeDetector()
        self.sign_detector = SignDetector()
        self.lane_detector = LaneDetector()
        self.behavior_monitor = IMUBehavior()
        self.risk_engine = RiskEngine()
        self.pothole_logger = PotholeMapLogger(SAFETY_DB_PATH)
        video_path = source if isinstance(source, str) and os.path.isfile(source) else None
        self.gps = GPSReader(video_path=video_path)
        self.kalman = RiskKalmanFilter()
        self.alerter = AudioAlerter()

        # State
        self.current_speed = 0
        self.is_running = True
        self.frame_count = 0
        
        self._init_dbs()

    def _init_dbs(self):
        # Main Events DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                type TEXT,
                value REAL,
                details TEXT,
                vehicle_id TEXT,
                image_path TEXT,
                risk_level TEXT
            )
        """)
        # Driver Behavior DB
        cur.execute("""
            CREATE TABLE IF NOT EXISTS driver_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                score REAL,
                speed REAL,
                risk_level TEXT
            )
        """)
        conn.commit()
        conn.close()

    def estimate_speed(self, frame):
        import random
        # Simple simulation: accelerate to 50, fluctuate
        if self.current_speed < 50:
            self.current_speed += 0.5
        
        if random.random() < 0.1:
            self.current_speed += random.uniform(-2, 2)
        return max(0, self.current_speed)

    def log_behavior(self, score, speed, risk_level):
        """Log driver behavior stats periodically"""
        if self.frame_count % 30 == 0: # Log every ~1 sec (assuming 30fps)
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("INSERT INTO driver_behavior (ts, score, speed, risk_level) VALUES (?, ?, ?, ?)",
                       (datetime.now().isoformat(), score, speed, risk_level))
            conn.commit()
            conn.close()

    def log_event(self, event_type, value, details, frame, risk_level="LOW"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{event_type}_{timestamp}.jpg"
        filepath = os.path.join(EVIDENCE_DIR, filename)
        cv2.imwrite(filepath, frame)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO events_v2 (ts, type, value, details, vehicle_id, image_path, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), event_type, value, details, "DEMO-CAR-01", filename, risk_level))
        conn.commit()
        conn.close()
        logger.info(f"Logged Event: {event_type} ({risk_level})")
        
        # Audio Alert for severe events
        if risk_level in ["HIGH", "CRITICAL"]:
            self.alerter.alert(f"Warning: {event_type} detected. Risk {risk_level}")

    def run(self):
        logger.info("System Started. Press 'q' to quit.")
        
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frame = cv2.resize(frame, (1020, 600))
            
            # --- VISION PIPELINE ---
            road_mask = self.road_segmenter.segment(frame)
            
            hazards = self.hazard_detector.detect(frame)
            hazards = self.road_segmenter.filter_detections(hazards, road_mask)
            
            potholes = self.pothole_detector.detect(frame)
            potholes = self.road_segmenter.filter_detections(potholes, road_mask)
            
            signs = self.sign_detector.detect(frame)
            current_limit = self.sign_detector.get_active_speed_limit()
            
            lane_status = self.lane_detector.detect(frame)
            
            # --- BEHAVIOR & SENSORS ---
            speed = self.estimate_speed(frame)
            # Simulate slight swerving effect for behavior score if speed is high
            behavior_score = self.behavior_monitor.get_behavior_score()
            if speed > 60: behavior_score -= 0.01 
            
            lat, lon = self.gps.get_location()

            # --- RISK FUSION ---
            raw_risk = self.risk_engine.compute(
                hazards, potholes, lane_status, speed, current_limit, behavior_score
            )
            # Kalman Filter Smoothing
            risk_score = self.kalman.update(raw_risk)
            # Re-classify level based on smoothed score
            risk_level = "LOW"
            if risk_score > 0.3: risk_level = "MEDIUM"
            if risk_score > 0.6: risk_level = "HIGH"
            if risk_score > 0.8: risk_level = "CRITICAL"

            # --- LOGGING & ALERTS ---
            self.log_behavior(behavior_score, speed, risk_level)

            for p in potholes:
                self.pothole_logger.log_pothole(lat, lon, p['severity'])
                if p['severity'] > 0.7:
                    self.log_event(EventType.POTHOLE, p['severity'], "Severe Pothole", frame, risk_level)

            for h in hazards:
                if h['severity'] > 0.6:
                     self.log_event(EventType.HAZARD, h['severity'], f"Hazard: {h['subtype']}", frame, risk_level)

            if lane_status.get('is_crossing') and speed > 10:
                deviation = lane_status.get('deviation', 0.0)
                self.log_event(EventType.LANE_DEPARTURE, deviation, "Lane Crossing", frame, risk_level)

            if current_limit and speed > current_limit + 5:
                # Debounce speeding alerts
                if self.frame_count % 60 == 0:
                     self.log_event(EventType.SPEEDING, speed, f"Limit: {current_limit}", frame, risk_level)

            # --- VISUALIZATION ---
            self._draw_overlay(frame, road_mask, hazards, potholes, signs, lane_status, risk_score, risk_level, speed, current_limit, behavior_score)
            
            cv2.imshow("AI Road Safety Platform", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False

        self.cap.release()
        cv2.destroyAllWindows()

    def _draw_overlay(self, frame, mask, hazards, potholes, signs, lane_status, risk, level, speed, limit, behavior):
        # Draw Road Mask (Transparent Blue)
        # overlay = frame.copy()
        # overlay[mask > 0] = (255, 0, 0) # This might be slow doing pixel assignment
        # cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Boxes
        for h in hazards:
             x1, y1, x2, y2 = map(int, h['bbox'])
             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
             cv2.putText(frame, f"Hazard {h['severity']:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for p in potholes:
             x1, y1, x2, y2 = map(int, p['bbox'])
             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
             cv2.putText(frame, "Pothole", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for s in signs:
             x1, y1, x2, y2 = map(int, s['bbox'])
             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
             cv2.putText(frame, s['subtype'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lane
        if lane_status.get('left_x'):
             cv2.line(frame, (int(lane_status['left_x']), frame.shape[0]), (int(lane_status['left_x']), int(frame.shape[0]*0.6)), (255, 255, 0), 3)

        # HUD
        h, w = frame.shape[:2]
        
        # Risk Bar
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        
        color = (0, 255, 0)
        if level == "MEDIUM": color = (0, 255, 255)
        if level == "HIGH": color = (0, 165, 255)
        if level == "CRITICAL": color = (0, 0, 255)
        
        cv2.putText(frame, f"RISK: {level}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Risk Meter Bar
        bar_width = 200
        cv2.rectangle(frame, (250, 20), (250 + bar_width, 50), (100, 100, 100), 2)
        cv2.rectangle(frame, (252, 22), (250 + int(risk * bar_width), 48), color, -1)
        
        cv2.putText(frame, f"Driver Score: {behavior:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, f"SPD: {int(speed)} km/h", (w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if limit:
             cv2.putText(frame, f"LIM: {limit}", (w - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="Video source (0/1 for webcam, or path to file)")
    args = parser.parse_args()
    
    src = args.source
    if str(src).isdigit():
        src = int(src)
        
    system = RoadSafetySystem(src)
    system.run()
