# api_server.py - Unified backend with in-process MJPEG streaming
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import sqlite3
import os
import sys
import threading
import time
import hashlib
import secrets
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
from functools import wraps
import json
import shutil
import tempfile
from collections import defaultdict, deque
import subprocess

# ── Path setup ──────────────────────────────────────────────────────────────
API_DIR      = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.dirname(API_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Model imports (loaded eagerly at startup for Railway) ─────────────────────
def _load_models():
    global hazard_model, pothole_model, sign_model, lane_model, road_model
    global speed_sign_model
    global MODELS_LOADED

    if MODELS_LOADED:
        return

    import torch
    from ultralytics import YOLO

    # Wrap each model in try/except so one failure doesn't kill the rest
    try:
        from perception.hazard_detector import HazardDetector
        print("[LOADING] Hazard model...")
        hazard_model = HazardDetector(model_path=os.path.join(BACKEND_DIR, "models/hazard_model/best.pt"))
        print("[OK] Hazard model loaded")
    except Exception as e:
        print(f"[FAIL] Hazard model: {e}")

    try:
        from perception.pothole_detector import PotholeDetector
        print("[LOADING] Pothole model...")
        pothole_model = PotholeDetector(model_path=os.path.join(BACKEND_DIR, "models/pothole_model/best.pt"))
        print("[OK] Pothole model loaded")
    except Exception as e:
        print(f"[FAIL] Pothole model: {e}")

    try:
        from perception.sign_detector import SignDetector
        print("[LOADING] Sign model...")
        sign_model = SignDetector(model_path=os.path.join(BACKEND_DIR, "models/sign_model/bestS.pt"))
        print("[OK] Sign model loaded")
    except Exception as e:
        print(f"[FAIL] Sign model: {e}")

    try:
        from perception.lane_detector import LaneDetector
        print("[LOADING] Lane model...")
        lane_model = LaneDetector(model_path=os.path.join(BACKEND_DIR, "models/lane_model/lane_detector.pth"))
        print("[OK] Lane model loaded")
    except Exception as e:
        print(f"[FAIL] Lane model: {e}")

    try:
        from perception.road_segmenter import RoadSegmenter
        print("[LOADING] Road segmenter...")
        road_model = RoadSegmenter(model_path=os.path.join(BACKEND_DIR, "models/road_segmentation/best.pth"))
        print("[OK] Road segmenter loaded")
    except Exception as e:
        print(f"[FAIL] Road segmenter: {e}")

    MODELS_LOADED = True

    # Load Piranesh's speed/junction YOLO model (separate from friend's sign model)
    try:
        from sign_detector import YOLOSignDetector
        _speed_weights = os.path.join(BACKEND_DIR, "runs", "detect", "runs", "detect",
                                       "speed_junction_v1", "weights", "best.pt")
        if os.path.exists(_speed_weights):
            speed_sign_model = YOLOSignDetector(_speed_weights, conf=0.45, imgsz=640)
            print("[OK] Speed/Junction sign model loaded")
        else:
            print(f"[WARN] Speed sign weights not found: {_speed_weights}")
    except Exception as e:
        print(f"[WARN] Could not load speed sign model: {e}")

    loaded = sum(1 for m in [hazard_model, pothole_model, sign_model, lane_model, road_model] if m is not None)
    print(f"[READY] {loaded}/5 models loaded successfully.")

# Global model handles (None until loaded)
hazard_model  = None
pothole_model = None
sign_model    = None
speed_sign_model = None   # Piranesh's speed/junction detector
lane_model    = None
road_model    = None
MODELS_LOADED = False

# ── Cloudinary helper for persistent evidence storage ─────────────────────────
try:
    from utils.cloudinary_helper import upload_evidence_image, get_evidence_url, is_cloudinary_active
    print("[STARTUP] Cloudinary helper loaded")
except ImportError as _cld_err:
    print(f"[WARN] Cloudinary helper not available: {_cld_err}")
    def upload_evidence_image(*a, **kw): return None
    def get_evidence_url(*a, **kw): return None
    def is_cloudinary_active(): return False


# ── Piranesh's geometric filter for speed signs ──────────────────────────────
def verify_speed_sign_circle(frame, x1, y1, x2, y2, cls_name, conf=0.5):
    """
    Rejects diamond/square shapes. Only accepts circular signs.
    ONLY call this for Speed-XX detections, NOT for other models.
    Returns True = real speed sign, False = false positive (diamond/warning sign)
    """
    import numpy as np
    h, w = frame.shape[:2]

    # Expand bounding box by 20% to include red border
    bw, bh = x2 - x1, y2 - y1
    expand = int(max(bw, bh) * 0.20)
    x1 = max(0, int(x1) - expand)
    y1 = max(0, int(y1) - expand)
    x2 = min(w, int(x2) + expand)
    y2 = min(h, int(y2) + expand)

    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 25 or roi_w < 25:
        return True  # Too small to check, let it through

    # STEP 1: CIRCULARITY CHECK (with Gaussian blur for close-up signs)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Smooth noise for close-up signs
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity = 1.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0 and area > 100:
            circularity = (4 * 3.14159 * area) / (perimeter * perimeter)

    # High-confidence YOLO detections get a more lenient threshold
    # True diamond shapes have circularity ~0.05-0.15
    # Close-up circular signs sometimes get 0.20-0.35 due to binarization artifacts
    circ_threshold = 0.25 if conf >= 0.85 else 0.40

    if circularity < circ_threshold:
        print(f"[CIRCLE-FILTER] {cls_name} REJECTED: Circularity={circularity:.2f} conf={conf:.2f} (diamond shape)")
        return False

    # For high-confidence detections that pass circularity, trust YOLO
    if conf >= 0.85 and circularity >= 0.25:
        print(f"[CIRCLE-FILTER] {cls_name} PASSED: Circularity={circularity:.2f} conf={conf:.2f} (high-conf)")
        return True

    # STEP 2: RED BORDER CHECK (for lower-confidence detections)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 80, 70]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([165, 80, 70]), np.array([180, 255, 255]))
    )
    red_ratio = cv2.countNonZero(mask_red) / (roi_h * roi_w)

    if circularity >= 0.55:
        if red_ratio < 0.005 and roi_w > 15:
            return False  # Round but no red = not a speed sign
        return True  # Round + has red = real speed sign

    # Ambiguous zone: need at least 2% red
    if red_ratio < 0.02:
        return False
    return True

# ── Streaming state ──────────────────────────────────────────────────────────
latest_frame   = None       # latest annotated JPEG bytes
latest_raw_frame = None
stream_lock    = threading.Lock()
stream_running = False
stream_thread  = None
road_monitor_running = False
road_monitor_stats = {}     # shared from road_monitor.py pipeline
stream_stats   = {          # sent to frontend via /api/stream/status
    "risk": 0.0,
    "risk_level": "IDLE",
    "hazards": 0,
    "potholes": 0,
    "signs": [],
    "lane_crossing": False,
    "speed_limit": None,
    "frame": 0,
    "source": "—",
    "location": None,         # (lat, lon) extracted from video, or None
}

DB_PATH       = os.path.join(BACKEND_DIR, "db/new.db")
EVIDENCE_DIR  = os.path.join(BACKEND_DIR, "evidence")
UPLOAD_FOLDER = os.path.join(BACKEND_DIR, "uploads")
POTHOLE_DB    = os.path.join(BACKEND_DIR, "database", "safety_events.db")
HAZARD_DB     = os.path.join(BACKEND_DIR, "database", "hazard_events.db")
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(POTHOLE_DB), exist_ok=True)
os.makedirs(os.path.dirname(HAZARD_DB), exist_ok=True)


def extract_video_gps(filepath):
    """
    Extract GPS coordinates embedded in video metadata.
    Tries ffprobe first, then falls back to a pure-Python MP4 box parser
    that reads the ©xyz atom (ISO 6709 short format) directly from the file.
    Returns (lat, lon) tuple of floats, or None if not found.
    """
    import re

    # ── Attempt 1: ffprobe ───────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', '-show_streams', filepath],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            all_tags = dict(data.get('format', {}).get('tags', {}))
            for stream in data.get('streams', []):
                all_tags.update(stream.get('tags', {}))
            tags = {k.lower(): v for k, v in all_tags.items()}
            for key in ('location', 'com.apple.quicktime.location.iso6709',
                        'gps', 'geo.location', 'location-eng', 'xyz'):
                val = tags.get(key)
                if not val:
                    continue
                m = re.search(r'([+-]\d{1,3}\.?\d*)([+-]\d{1,3}\.?\d*)', val)
                if m:
                    lat, lon = float(m.group(1)), float(m.group(2))
                    if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0):
                        print(f"[GPS] ffprobe: lat={lat}, lon={lon} (tag={key})")
                        return (lat, lon)
    except FileNotFoundError:
        pass   # ffprobe not installed — fall through to pure-Python parser
    except Exception:
        pass

    # ── Attempt 2: pure-Python MP4 ©xyz box parser ───────────────────────────
    try:
        import struct
        with open(filepath, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            pos = 0
            while pos < file_size:
                f.seek(pos)
                header = f.read(8)
                if len(header) < 8:
                    break
                size = struct.unpack('>I', header[:4])[0]
                name = header[4:8]
                if size == 0:
                    size = file_size - pos
                elif size == 1:
                    ext = f.read(8)
                    size = struct.unpack('>Q', ext)[0]
                if name == b'moov':
                    f.seek(pos)
                    moov_data = f.read(size)
                    # ©xyz is stored as 0xa9 0x78 0x79 0x7a
                    idx = moov_data.find(b'\xa9xyz')
                    if idx != -1:
                        raw = moov_data[idx + 8: idx + 58].decode('latin1', errors='replace')
                        m = re.search(r'([+-]\d+\.\d+)([+-]\d+\.\d+)', raw)
                        if m:
                            lat, lon = float(m.group(1)), float(m.group(2))
                            if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0):
                                print(f"[GPS] ©xyz box: lat={lat}, lon={lon}")
                                return (lat, lon)
                    break
                pos += size
    except Exception as e:
        print(f"[GPS] Pure-Python extraction error: {e}")

    return None


def _init_pothole_db():
    """Ensure the potholes map table exists."""
    conn = sqlite3.connect(POTHOLE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL, longitude REAL,
            severity REAL, timestamp REAL,
            vehicle_id TEXT, source TEXT, image_path TEXT
        )
    """)
    # backwards-compat: add new columns if missing
    for col in ("vehicle_id TEXT", "source TEXT", "image_path TEXT"):
        try:
            conn.execute(f"ALTER TABLE potholes ADD COLUMN {col}")
        except Exception:
            pass
    conn.commit(); conn.close()


def _init_hazard_db():
    """Ensure a dedicated hazards table exists."""
    conn = sqlite3.connect(HAZARD_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hazards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            hazard_type TEXT,
            severity REAL,
            confidence REAL,
            risk_level TEXT,
            details TEXT,
            vehicle_id TEXT,
            source TEXT,
            latitude REAL,
            longitude REAL,
            image_path TEXT
        )
    """)
    for col in (
        "hazard_type TEXT",
        "severity REAL",
        "confidence REAL",
        "risk_level TEXT",
        "details TEXT",
        "vehicle_id TEXT",
        "source TEXT",
        "latitude REAL",
        "longitude REAL",
        "image_path TEXT",
    ):
        try:
            conn.execute(f"ALTER TABLE hazards ADD COLUMN {col}")
        except Exception:
            pass
    conn.commit(); conn.close()


_init_pothole_db()
_init_hazard_db()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, "frontend"))
CORS(app, resources={r"/api/*": {"origins": "*"}},
     supports_credentials=False,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Register inference blueprint ───────────────────────────────────────────────
try:
    from api.inference_routes import inference_bp
    app.register_blueprint(inference_bp)
    print("[SUCCESS] Registered inference blueprint")
except ImportError as e:
    print(f"[WARNING] Could not register inference blueprint: {e}")

# ── Eager model loading (load at startup so Railway has them ready) ────────────
def _eager_load_models():
    """Load all ML models in a background thread at server startup."""
    import time
    time.sleep(2)  # Let gunicorn finish binding first
    print("[STARTUP] Loading ML models eagerly...")
    try:
        _load_models()
    except Exception as e:
        print(f"[STARTUP] Model loading error (non-fatal): {e}")

_model_loader_thread = threading.Thread(target=_eager_load_models, daemon=True)
_model_loader_thread.start()
print("[STARTUP] Model loading thread started")

# ── Token store ───────────────────────────────────────────────────────────────
tokens = {}  # token -> {user_id, role, expires}

def get_current_user():
    """Get current user from Authorization header token."""
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        token = auth[7:]
        info = tokens.get(token)
        if info and datetime.fromisoformat(info['expires']) > datetime.now():
            return info
    return None

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        if user.get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist + default admin."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        name TEXT,
        role TEXT DEFAULT 'user',
        created_at TEXT,
        updated_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS events_v2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, type TEXT, value REAL, details TEXT,
        vehicle_id TEXT, image_path TEXT, risk_level TEXT
    )""")
    # Add risk_level column if missing (backwards compat)
    try: cur.execute("ALTER TABLE events_v2 ADD COLUMN risk_level TEXT")
    except: pass
    cur.execute("""CREATE TABLE IF NOT EXISTS vehicles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id TEXT UNIQUE,
        owner_id INTEGER,
        label TEXT,
        speed_limit_kmh REAL,
        lane_sensitivity_px REAL,
        notes TEXT,
        kmh_per_pixel REAL,
        created_at TEXT,
        updated_at TEXT,
        FOREIGN KEY (owner_id) REFERENCES users(id)
    )""")
    try: cur.execute("ALTER TABLE vehicles ADD COLUMN owner_id INTEGER")
    except: pass
    try: cur.execute("ALTER TABLE vehicles ADD COLUMN kmh_per_pixel REAL")
    except: pass
    cur.execute("""CREATE TABLE IF NOT EXISTS driver_behavior (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, behavior TEXT, score REAL,
        latitude REAL, longitude REAL, metadata TEXT
    )""")
    # Add geolocation columns if missing
    for col in ["latitude REAL", "longitude REAL", "metadata TEXT"]:
        try:
            cur.execute(f"ALTER TABLE driver_behavior ADD COLUMN {col}")
        except Exception:
            pass
    # Default admin
    cur.execute("SELECT id FROM users WHERE email='admin@system.local' OR name='Admin'")
    if not cur.fetchone():
        cur.execute("INSERT INTO users (email, password_hash, name, role, created_at) VALUES (?,?,?,?,?)",
                    ('admin@system.local', hashlib.sha256('Admin'.encode()).hexdigest(), 'Admin', 'admin', datetime.now().isoformat()))
    conn.commit()
    # Seed demo vehicles for admin (owner_id=1) if none exist
    cur.execute("SELECT id FROM users WHERE email='admin@system.local' OR name='Admin'")
    admin_row = cur.fetchone()
    if admin_row:
        aid = admin_row[0]
        cur.execute("SELECT COUNT(*) FROM vehicles WHERE owner_id=?", (aid,))
        if cur.fetchone()[0] == 0:
            now_s = datetime.now().isoformat()
            seeds = [
                ('DEMO-CAR-01', aid, 'Demo Vehicle',    70, 'Default demo vehicle'),
                ('BUS_01',      aid, 'School Bus',       60, 'Test bus'),
                ('CAR_01',      aid, 'Demo Car',         80, 'Test car'),
                ('VAN_01',      aid, 'Delivery Van',     60, 'Test van'),
            ]
            for vid, oid, lbl, spd, notes in seeds:
                try:
                    cur.execute(
                        "INSERT INTO vehicles (vehicle_id,owner_id,label,speed_limit_kmh,notes,created_at) VALUES (?,?,?,?,?,?)",
                        (vid, oid, lbl, spd, notes, now_s))
                except Exception:
                    pass
    conn.commit(); conn.close()
    print("[DB] Tables initialized.")

init_db()

# Pre-load models in background so first upload is faster
threading.Thread(target=_load_models, daemon=True).start()

def _capture_frames(cap):
    global latest_raw_frame, stream_running

    while stream_running and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))
        
# ── Detection thread ──────────────────────────────────────────────────────────
def _process_stream(source, vehicle_id='DEMO-CAR-01', video_gps=None):
    global latest_frame, stream_running, stream_stats, latest_raw_frame

    if video_gps:
        stream_stats['location'] = list(video_gps)
        print(f"[STREAM] GPS for this video: lat={video_gps[0]}, lon={video_gps[1]}")

    _load_models()

    # Reset pothole detector Kalman map for this new stream
    if pothole_model is not None and hasattr(pothole_model, 'reset'):
        pothole_model.reset()
    import torch
    from utils.kalman_filter import RiskKalmanFilter
    kalman = RiskKalmanFilter()

    try:
        cap = cv2.VideoCapture(source)
        # capture_thread = threading.Thread(
        #     target=_capture_frames,
        #     args=(cap,),
        #     daemon=True
        # )
        # capture_thread.start()
    except Exception as e:
        print(f"[STREAM] Cannot open source: {e}")
        stream_running = False
        return

    if not cap.isOpened():
        print(f"[STREAM] Cannot open: {source}")
        stream_running = False
        return

    ALPHA = 0.65
    prev_risk = 0.0
    frame_n   = 0
    PERSIST_FRAMES = 3
    track_history = defaultdict(lambda: deque(maxlen=PERSIST_FRAMES))
    road_mask_cache = None
    last_log = 0.0
    last_det_log = 0.0

    BOTTOM_GATE    = 0.65
    ROAD_OVERLAP_TH = 0.15
    GREEN_TH = 0.20
    ORANGE_TH = 0.45
    # Cached results for frame skipping
    last_hazards = []
    last_potholes = []
    last_signs = []
    last_speed_signs = []   # Piranesh's speed/junction detections
    last_lane_status = {}
    pothole_log_count = 0   # how many potholes logged to map this stream

    src_label = str(source) if isinstance(source, str) else "Live Camera"
    print(f"[STREAM] Starting: {src_label}")

    try:
        while stream_running and cap.isOpened():
            t_start = time.time()
            ret, frame = cap.read()

            if not ret:
                print(f"[STREAM] End of video or read error at frame {frame_n}", flush=True)
                break

            frame = cv2.resize(frame, (960, 540))
            frame_n += 1
            h, w = frame.shape[:2]

            # ── Road mask (every 3 frames) ──────────────────────────────────
            if frame_n % 6 == 0 or road_mask_cache is None:
                t_rm = time.time()
                road_mask_cache = road_model.segment(frame)
                if frame_n % 30 == 0:
                    print(f"[STREAM] Road model seg took {time.time()-t_rm:.3f}s", flush=True)
            road_mask = road_mask_cache

            # ── Draw semi-transparent road mask ─────────────────────────────
            if road_mask is not None:
                overlay = frame.copy()
                overlay[road_mask > 0] = (
                    overlay[road_mask > 0] * 0.6 +
                    np.array([180, 120, 50], dtype=np.uint8) * 0.4
                ).astype(np.uint8)
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            # ── Hazard detection (every 3 frames) ─────────────────────────────
            if frame_n % 3 == 0:
                hazards = hazard_model.detect(frame)
                hazards = road_model.filter_detections(hazards, road_mask, threshold=0.15)
                last_hazards = hazards
            else:
                hazards = last_hazards

            # ── Pothole detection (every 3 frames) ────────────────────────────
            if frame_n % 3 == 0:
                potholes = pothole_model.detect(frame)
                potholes = road_model.filter_detections(potholes, road_mask, threshold=0.15)
                last_potholes = potholes
            else:
                potholes = last_potholes

            # ── Sign detection (every 4 frames) ───────────────────────────────
            if frame_n % 2 == 0:

                signs = sign_model.detect(frame)

                # Filter signs near road region
                signs = road_model.filter_detections(
                    signs,
                    road_mask,
                    threshold=0.05
                )

                last_signs = signs

            else:
                signs = last_signs


            # ── Speed/Junction sign detection (Piranesh's model, every 4 frames)
            if frame_n % 4 == 0 and speed_sign_model is not None:
                raw_speed_signs = speed_sign_model.detect(frame)
                # Apply circle filter ONLY to Speed-XX classes (reject diamond signs)
                speed_signs = []
                for det in raw_speed_signs:
                    cls_name = det['cls_name']
                    if cls_name.startswith('Speed-'):
                        x1, y1, x2, y2 = det['xyxy']
                        if not verify_speed_sign_circle(frame, x1, y1, x2, y2, cls_name, conf=det['conf']):
                            continue  # Diamond shape — skip
                    speed_signs.append(det)
                last_speed_signs = speed_signs
            else:
                speed_signs = last_speed_signs
            
            # Default speed limit since road_monitor thread handles it now
            speed_limit = None

            # ── Speed/Junction sign detection (Piranesh's model, every 4 frames)
            if frame_n % 4 == 0 and speed_sign_model is not None:
                raw_speed_signs = speed_sign_model.detect(frame)
                # Apply circle filter ONLY to Speed-XX classes (reject diamond signs)
                speed_signs = []
                for det in raw_speed_signs:
                    cls_name = det['cls_name']
                    if cls_name.startswith('Speed-'):
                        x1, y1, x2, y2 = det['xyxy']
                        if not verify_speed_sign_circle(frame, x1, y1, x2, y2, cls_name, conf=det['conf']):
                            continue  # Diamond shape — skip
                    speed_signs.append(det)
                last_speed_signs = speed_signs
            else:
                speed_signs = last_speed_signs

            # ── Lane detection (every 3 frames) ───────────────────────────────
            if frame_n % 3 == 0:
                lane_status = lane_model.detect(frame)
                last_lane_status = lane_status
            else:
                lane_status = last_lane_status

            # ── Risk computation ─────────────────────────────────────────────
            img_h = frame.shape[0]

            haz_score = 0.0

            for det in hazards:

                x1, y1, x2, y2 = det['bbox']
                x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)

                raw_bottom = y2 / img_h
                bottom_position = max(0.0, (raw_bottom - BOTTOM_GATE) / (1.0 - BOTTOM_GATE))
                bottom_position = min(bottom_position, 1.0)

                risk_box = bottom_position

                key = (x1 // 80, y1 // 80)
                track_history[key].append(risk_box)

                # Require persistence
                if len(track_history[key]) < PERSIST_FRAMES:
                    continue

                haz_score = max(haz_score, risk_box)

                # --- Object risk visualization ---
                if risk_box >= ORANGE_TH:
                    color = (0,0,255)
                    label = "HIGH RISK"

                elif risk_box >= GREEN_TH:
                    color = (0,165,255)
                    label = "MEDIUM RISK"

                else:
                    color = (0,255,0)
                    label = "LOW RISK"

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                cv2.putText(frame, label,
                            (x1, max(y1-8,12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2)
            pot_score = min(
                1.0,
                sum(p['severity'] for p in potholes)
            )
            lane_score = 1.0 if lane_status.get('is_crossing') else 0.0
            raw_risk   = min(0.35*haz_score + 0.35*pot_score + 0.15*lane_score + 0.15*prev_risk, 1.0)
            risk       = kalman.update(raw_risk)
            prev_risk  = ALPHA * prev_risk + (1 - ALPHA) * risk

            if risk > 0.8:   risk_level, rc = "CRITICAL", (0, 0, 255)
            elif risk > 0.6: risk_level, rc = "HIGH",     (0, 80, 255)
            elif risk > 0.3: risk_level, rc = "MEDIUM",   (0, 200, 255)
            else:            risk_level, rc = "LOW",       (0, 220, 80)

            # ── Draw detections ──────────────────────────────────────────────

            for det in potholes:
                x1,y1,x2,y2 = map(int, det['bbox'])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
                cv2.putText(frame, f"Pothole {det['severity']:.2f}",
                            (x1, max(y1-6,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)

            for det in signs:
                x1,y1,x2,y2 = map(int, det['bbox'])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,80), 2)
                cv2.putText(frame, det['subtype'],
                            (x1, max(y1-6,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,80), 2)

            # Draw Piranesh's speed/junction sign detections
            if speed_sign_model is not None:
                frame = speed_sign_model.draw(frame, speed_signs)

            # Lane mask overlay
            frame = lane_model.draw_lanes(frame, lane_status)

            # ── HUD ──────────────────────────────────────────────────────────
            # Top bar (clean header without risk text)
            cv2.rectangle(frame, (0,0), (w, 50), (15,15,20), -1)

            # Detection counts
            info = f"Hazards:{len(hazards)}  Potholes:{len(potholes)}  Signs:{len(signs)}"
            cv2.putText(frame, info, (w//2-180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

            cv2.putText(frame, "AI ROAD SAFETY SYSTEM",
                        (w-220, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)

            # ── Draw road_monitor overlay ───────────────────────────────────
            if road_monitor_stats:
                s = road_monitor_stats
                try:
                    speed_val   = float(s.get('speed_kmh') or 0.0)
                    limit_val   = float(s.get('active_limit') or 0.0)
                    color       = (0, 255, 0) if speed_val <= limit_val else (0, 0, 255)
                    speed_str   = f"{speed_val:.1f}"
                    limit_str   = f"{limit_val:.1f}" if limit_val > 0 else "---"

                    cv2.putText(frame, f"Speed: {speed_str} km/h | Limit: {limit_str} km/h",
                                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if s.get('is_junction'):
                        cv2.putText(frame, "JUNCTION: JUNCTION AHEAD", (20, 125),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                    
                    admin_limit_str = str(s.get('admin_limit')) if s.get('admin_limit') else "---"
                    cv2.putText(frame, f"LaneSens(px): {s.get('lane_sens')} | AdminLimit: {admin_limit_str}",
                                (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                except Exception as e:
                    print(f"HUD DRAW ERROR: {e}")
                    pass

            # ── Encode & share frame ─────────────────────────────────────────
            # Ensure the stream handler (which reads latest_frame) gets the newly drawn frame
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
            with stream_lock:
                latest_frame = buf.tobytes()
                stream_stats.update({
                    "risk": round(float(risk), 3),
                    "risk_level": risk_level,
                    "hazards": len(hazards),
                    "potholes": len(potholes),
                    "signs": [s['subtype'] for s in signs],
                    "lane_crossing": bool(lane_status.get('is_crossing')),
                    "speed_limit": speed_limit,
                    "frame": frame_n,
                    "source": src_label,
                    "risk_components": {
                        "hazard": float(haz_score),
                        "pothole": float(pot_score),
                        "lane": float(lane_score),
                        "temporal": float(prev_risk)
                    }
                })

            # ── DB log ──────────────────────────────────────────────────────
            now = time.time()
            if risk > 0.6 and now - last_log > 3.0:
                last_log = now
                ts  = datetime.now()
                fn  = f"HIGH_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(os.path.join(EVIDENCE_DIR, fn), frame)
                try:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute(
                        "INSERT INTO events_v2 (ts,type,value,details,vehicle_id,image_path,risk_level) VALUES (?,?,?,?,?,?,?)",
                        (ts.isoformat(), "HIGH_RISK", float(risk), risk_level, vehicle_id, fn, risk_level)
                    )
                    conn.commit(); conn.close()
                except Exception as dbe:
                    print(f"[DB] {dbe}")

            # ── Log potholes to map DB when GPS is available ──────────────────
            if video_gps and len(potholes) > 0 and now - last_det_log > 8.0:
                import random
                base_lat, base_lon = video_gps
                try:
                    # Save one evidence frame for this GPS pothole batch
                    ts_ph  = datetime.now()
                    ev_fn  = f"PH_{ts_ph.strftime('%Y%m%d_%H%M%S')}.jpg"
                    # Try Cloudinary first, fall back to local
                    if not upload_evidence_image(frame, ev_fn):
                        cv2.imwrite(os.path.join(EVIDENCE_DIR, ev_fn), frame)
                    pconn = sqlite3.connect(POTHOLE_DB)
                    for p in potholes:
                        # Small random jitter so each detection appears as a
                        # distinct point (~±20 m) rather than stacking.
                        jitter_lat = random.uniform(-0.0002, 0.0002)
                        jitter_lon = random.uniform(-0.0002, 0.0002)
                        pconn.execute(
                            "INSERT INTO potholes (latitude,longitude,severity,timestamp,vehicle_id,source,image_path) VALUES (?,?,?,?,?,?,?)",
                            (base_lat + jitter_lat, base_lon + jitter_lon,
                             float(p['severity']), time.time(), vehicle_id, src_label, ev_fn)
                        )
                        pothole_log_count += 1
                    pconn.commit(); pconn.close()
                    print(f"[MAP] Logged {len(potholes)} pothole(s) with evidence (total {pothole_log_count})")
                except Exception as pe:
                    print(f"[MAP] Pothole DB error: {pe}")

            # ── DB log detections (rate-limited, every 5 s) ──────────────────
            if (len(hazards) > 0 or len(potholes) > 0 or len(signs) > 0) and now - last_det_log > 5.0:
                last_det_log = now
                try:
                    conn = sqlite3.connect(DB_PATH)
                    ts_d   = datetime.now()
                    ts_iso = ts_d.isoformat()
                    if len(hazards) > 0:
                        hz_fn = f"HZ_{ts_d.strftime('%Y%m%d_%H%M%S')}.jpg"
                        # Try Cloudinary first, fall back to local
                        if not upload_evidence_image(frame, hz_fn):
                            cv2.imwrite(os.path.join(EVIDENCE_DIR, hz_fn), frame)
                        avg_hz_sev = float(sum(h.get('severity', 0.0) for h in hazards) / max(len(hazards), 1))
                        avg_hz_conf = float(sum(h.get('confidence', 0.0) for h in hazards) / max(len(hazards), 1))
                        hz_details = str([h.get('subtype', 'unknown') for h in hazards])
                        conn.execute(
                            "INSERT INTO events_v2 (ts,type,value,details,vehicle_id,image_path,risk_level) VALUES (?,?,?,?,?,?,?)",
                            (ts_iso, "hazard", avg_hz_sev,
                             hz_details, vehicle_id, hz_fn, risk_level)
                        )
                        try:
                            hconn = sqlite3.connect(HAZARD_DB)
                            hlat, hlon = (video_gps if video_gps else (None, None))
                            for hz in hazards:
                                hconn.execute(
                                    """
                                    INSERT INTO hazards (ts,hazard_type,severity,confidence,risk_level,details,vehicle_id,source,latitude,longitude,image_path)
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                                    """,
                                    (
                                        ts_iso,
                                        str(hz.get('subtype', 'unknown')),
                                        float(hz.get('severity', 0.0)),
                                        float(hz.get('confidence', 0.0)),
                                        risk_level,
                                        hz_details,
                                        vehicle_id,
                                        src_label,
                                        hlat,
                                        hlon,
                                        hz_fn,
                                    )
                                )
                            hconn.commit(); hconn.close()
                        except Exception as hz_db_e:
                            print(f"[DB HAZARD] {hz_db_e}")
                    if len(potholes) > 0:
                        # Save evidence frame for this pothole detection
                        ph_fn   = f"PH_{ts_d.strftime('%Y%m%d_%H%M%S')}.jpg"
                        # Try Cloudinary first, fall back to local
                        if not upload_evidence_image(frame, ph_fn):
                            cv2.imwrite(os.path.join(EVIDENCE_DIR, ph_fn), frame)
                        max_sev = max(p['severity'] for p in potholes)
                        conn.execute(
                            "INSERT INTO events_v2 (ts,type,value,details,vehicle_id,image_path,risk_level) VALUES (?,?,?,?,?,?,?)",
                            (ts_iso, "pothole", float(max_sev),
                             f"count={len(potholes)}", vehicle_id, ph_fn, risk_level)
                        )
                        # Also log to potholes map table (lat/lon=None when no GPS)
                        if not video_gps:
                            try:
                                pconn2 = sqlite3.connect(POTHOLE_DB)
                                pconn2.execute(
                                    "INSERT INTO potholes (latitude,longitude,severity,timestamp,vehicle_id,source,image_path) VALUES (?,?,?,?,?,?,?)",
                                    (None, None, float(max_sev), time.time(), vehicle_id, 'video_detection', ph_fn)
                                )
                                pconn2.commit(); pconn2.close()
                            except Exception as pe2:
                                print(f"[DB MAP] {pe2}")
                    if len(signs) > 0:
                        conn.execute(
                            "INSERT INTO events_v2 (ts,type,value,details,vehicle_id,risk_level) VALUES (?,?,?,?,?,?)",
                            (ts_iso, "sign", float(len(signs)),
                             str([s['subtype'] for s in signs]), vehicle_id, risk_level)
                        )
                    conn.commit(); conn.close()
                except Exception as dbe:
                    print(f"[DB DET] {dbe}")

            # Enforce playback speed (cap at ~30 FPS for recorded videos)
            elapsed = time.time() - t_start
            if frame_n % 30 == 0:
                print(f"[STREAM] Frame {frame_n} processed in {elapsed:.3f}s", flush=True)
            delay = max(0.0, (1.0 / 30.0) - elapsed)
            if delay > 0:
                time.sleep(delay)

    except Exception as ex:
        print(f"[STREAM ERROR] {ex}")
    finally:
        cap.release()
        stream_running = False
        print("[STREAM] Stopped.")


def _mjpeg_generator():
    global latest_frame
    while True:
        with stream_lock:
            frame_bytes = latest_frame
        if frame_bytes is None:
            # Send a black placeholder until stream starts
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for video source ...",
                        (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,100), 2)
            _, buf = cv2.imencode('.jpg', placeholder)
            frame_bytes = buf.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes + b'\r\n'
        )
        time.sleep(0.01)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/stream')
def stream():
    resp = Response(_mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # Explicit CORS for cross-origin <img> from Netlify frontend
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/api/stream/status')
def stream_status():
    with stream_lock:
        stats = dict(stream_stats)
    stats['running'] = stream_running
    return jsonify(stats)

@app.route('/api/stream/start', methods=['POST'])
def stream_start():
    global stream_running, stream_thread, latest_frame
    data   = request.get_json(force=True)
    source = data.get('source', 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if stream_running:
        return jsonify({'error': 'Already running. Stop first.'}), 400

    vehicle_id = data.get('vehicle_id', 'DEMO-CAR-01')
    latest_frame   = None
    stream_running = True
    stream_thread  = threading.Thread(target=_process_stream, args=(source, vehicle_id), daemon=True)
    stream_thread.start()
    return jsonify({'ok': True, 'source': str(source)})

@app.route('/api/stream/stop', methods=['POST'])
def stream_stop():
    global stream_running, latest_frame
    stream_running = False
    with stream_lock:
        latest_frame = None
    stream_stats.update({
        "risk": 0.0, "risk_level": "IDLE", "hazards": 0,
        "potholes": 0, "signs": [], "lane_crossing": False,
        "speed_limit": None, "frame": 0, "source": "—", "location": None,
    })
    return jsonify({'ok': True})

# Keep legacy /api/system/* aliases working
@app.route('/api/system/start', methods=['POST'])
def system_start():
    return stream_start()

@app.route('/api/system/stop', methods=['POST'])
def system_stop():
    return stream_stop()

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    global stream_running, stream_thread, latest_frame, road_monitor_running
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)

    # Get vehicle_id from form data (sent by analysis.html)
    vehicle_id = request.form.get('vehicle_id', 'DEMO-CAR-01')

    # Try to extract GPS from video metadata
    video_gps = extract_video_gps(filepath)
    gps_info = {'lat': video_gps[0], 'lon': video_gps[1]} if video_gps else None

    # Stop existing stream
    stream_running = False
    time.sleep(0.5)

    # Thread 1: MJPEG detection stream
    latest_frame   = None
    stream_running = True
    road_monitor_running = True
    stream_thread  = threading.Thread(target=_process_stream, args=(filepath, vehicle_id, video_gps), daemon=True)
    stream_thread.start()

    # Callback to receive stats (speed, lane limits) from road_monitor
    def _on_stats(stats):
        global road_monitor_stats
        road_monitor_stats = stats

    # Thread 2: road_monitor pipeline — runs headless, writes violations to DB
    def _run_road_monitor(video_path, vid):
        global road_monitor_running
        try:
            # Ensure PROJECT_ROOT is on path before importing road_monitor
            import sys as _sys
            if PROJECT_ROOT not in _sys.path:
                _sys.path.insert(0, PROJECT_ROOT)
            
            from road_monitor import run_monitor
            
            if not road_monitor_running:
                return
                
            print(f"[ROAD_MONITOR] Starting headless for vehicle={vid}", flush=True)
            run_monitor(video_path, vehicle_id=vid, headless=True, stats_callback=_on_stats)
            print(f"[ROAD_MONITOR] Finished for vehicle={vid}", flush=True)
        except Exception as e:
            import traceback
            print(f"[ROAD_MONITOR ERROR] {e}", flush=True)
            traceback.print_exc()

    rm_thread = threading.Thread(target=_run_road_monitor, args=(filepath, vehicle_id), daemon=True)
    rm_thread.start()

    return jsonify({
        'ok': True,
        'message': 'Uploaded & streaming! Violations being logged.',
        'path': filepath,
        'gps': gps_info,
    })

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json(force=True)
    u    = data.get('username') or data.get('email', '')
    pw   = data.get('password', '')
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT id,name,role,password_hash FROM users WHERE email=? OR name=?", (u, u))
    user = cur.fetchone(); conn.close()
    ph = hashlib.sha256(pw.encode()).hexdigest()
    if user and user['password_hash'] == ph:
        token = secrets.token_hex(32)
        from datetime import timedelta
        tokens[token] = {'user_id': user['id'], 'role': user['role'],
                         'expires': (datetime.now() + timedelta(hours=24)).isoformat()}
        return jsonify({'ok': True, 'token': token,
                        'role': user['role'], 'user_id': user['id'], 'name': user['name']})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json(force=True)
    ph   = hashlib.sha256((data.get('password','') or '').encode()).hexdigest()
    conn = get_db(); cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (email,name,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                    (data.get('email'), data.get('name'), ph, 'user', datetime.now().isoformat()))
        conn.commit(); uid = cur.lastrowid; conn.close()
        token = secrets.token_hex(32)
        from datetime import timedelta
        tokens[token] = {'user_id': uid, 'role': 'user',
                         'expires': (datetime.now() + timedelta(hours=24)).isoformat()}
        return jsonify({'ok': True, 'token': token, 'role': 'user', 'user_id': uid})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Email already exists'}), 400

@app.route('/api/auth/me')
@require_auth
def get_me():
    user = get_current_user()
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT id, email, name, role FROM users WHERE id=?", (user['user_id'],))
    row = cur.fetchone(); conn.close()
    if row:
        return jsonify(dict(row))
    return jsonify({'error': 'User not found'}), 404

# ── Data endpoints ────────────────────────────────────────────────────────────
@app.route('/api/user/events')
def get_events():
    conn = get_db(); cur = conn.cursor()
    ev_type = request.args.get('type')
    limit   = min(500, int(request.args.get('limit', 50)))
    if ev_type:
        cur.execute("SELECT * FROM events_v2 WHERE type=? ORDER BY ts DESC LIMIT ?", (ev_type, limit))
    else:
        cur.execute("SELECT * FROM events_v2 ORDER BY ts DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]; conn.close()
    return jsonify(rows)

@app.route('/api/potholes')
def get_potholes():
    if not os.path.exists(POTHOLE_DB): return jsonify([])
    conn = sqlite3.connect(POTHOLE_DB); conn.row_factory = sqlite3.Row; cur = conn.cursor()
    cur.execute("SELECT * FROM potholes ORDER BY timestamp DESC LIMIT 500")
    rows = [dict(r) for r in cur.fetchall()]; conn.close()
    return jsonify(rows)


@app.route('/api/hazards')
@require_auth
def get_hazards():
    limit = min(500, int(request.args.get('limit', 200)))
    conn = sqlite3.connect(HAZARD_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM hazards ORDER BY ts DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route('/api/potholes/add', methods=['POST'])
@require_auth
def pothole_add():
    """Drop a single pothole pin manually from the map."""
    import time as _time
    data = request.get_json(force=True)
    try:
        lat      = float(data['lat'])
        lon      = float(data['lon'])
        severity = max(0.0, min(1.0, float(data.get('severity', 0.5))))
        vehicle_id = data.get('vehicle_id', 'manual')
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'lat and lon required'}), 400
    _init_pothole_db()
    pconn = sqlite3.connect(POTHOLE_DB)
    cur = pconn.execute(
        "INSERT INTO potholes (latitude,longitude,severity,timestamp,vehicle_id,source) VALUES (?,?,?,?,?,?)",
        (lat, lon, severity, _time.time(), vehicle_id, 'map_drop')
    )
    new_id = cur.lastrowid
    pconn.commit(); pconn.close()
    return jsonify({'ok': True, 'id': new_id, 'lat': lat, 'lon': lon, 'severity': severity})


@app.route('/api/potholes/<int:pid>', methods=['DELETE', 'PATCH'])
@require_auth
def pothole_detail(pid):
    """Delete or update (severity) a pothole map point."""
    pconn = sqlite3.connect(POTHOLE_DB); pconn.row_factory = sqlite3.Row
    row = pconn.execute("SELECT * FROM potholes WHERE id=?", (pid,)).fetchone()
    if not row:
        pconn.close()
        return jsonify({'error': 'Not found'}), 404

    if request.method == 'DELETE':
        pconn.execute("DELETE FROM potholes WHERE id=?", (pid,))
        pconn.commit(); pconn.close()
        return jsonify({'ok': True})

    # PATCH — update severity and/or position
    data = request.get_json(force=True)
    severity = data.get('severity', row['severity'])
    lat      = data.get('lat',      row['latitude'])
    lon      = data.get('lon',      row['longitude'])
    try:
        severity = max(0.0, min(1.0, float(severity)))
        lat = float(lat); lon = float(lon)
    except (TypeError, ValueError):
        pconn.close()
        return jsonify({'error': 'Invalid values'}), 400
    pconn.execute("UPDATE potholes SET severity=?, latitude=?, longitude=? WHERE id=?",
                  (severity, lat, lon, pid))
    pconn.commit(); pconn.close()
    return jsonify({'ok': True})


@app.route('/api/potholes/manual', methods=['POST'])
@require_auth
def potholes_manual():
    """
    Pin potholes detected in the last stream onto the map using a
    manually-supplied GPS coordinate.  Takes the most recent unpinned
    pothole events for the given vehicle and assigns the supplied lat/lon.
    """
    data = request.get_json(force=True)
    lat = data.get('lat')
    lon = data.get('lon')
    vehicle_id = data.get('vehicle_id', 'DEMO-CAR-01')

    if lat is None or lon is None:
        return jsonify({'error': 'lat and lon are required'}), 400
    try:
        lat = float(lat); lon = float(lon)
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid lat/lon values'}), 400

    # Pull recent pothole events from violations.db for this vehicle
    import random, time as _time
    conn = get_db(); cur = conn.cursor()
    cur.execute(
        "SELECT value FROM events_v2 WHERE vehicle_id=? AND type='pothole' ORDER BY ts DESC LIMIT 20",
        (vehicle_id,)
    )
    pothole_rows = cur.fetchall()
    conn.close()

    if not pothole_rows:
        return jsonify({'ok': True, 'count': 0,
                        'message': 'No recent pothole events found for this vehicle.'})

    pconn = sqlite3.connect(POTHOLE_DB)
    count = 0
    for row in pothole_rows:
        severity = float(row[0]) if row[0] is not None else 0.5
        jitter_lat = random.uniform(-0.0002, 0.0002)
        jitter_lon = random.uniform(-0.0002, 0.0002)
        pconn.execute(
            "INSERT INTO potholes (latitude,longitude,severity,timestamp,vehicle_id,source) VALUES (?,?,?,?,?,?)",
            (lat + jitter_lat, lon + jitter_lon,
             min(severity, 1.0), _time.time(), vehicle_id, 'manual_pin')
        )
        count += 1
    pconn.commit(); pconn.close()

    print(f"[MAP] Manual GPS: pinned {count} pothole(s) at {lat},{lon} for {vehicle_id}")
    return jsonify({'ok': True, 'count': count})



# ── Vehicle Management (User) ─────────────────────────────────────────────────

@app.route('/api/user/vehicles', methods=['GET', 'POST'])
@require_auth
def user_vehicles():
    user = get_current_user()
    conn = get_db(); cur = conn.cursor()

    if request.method == 'GET':
        cur.execute("""SELECT v.*, 
            (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id) as total_events,
            (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id AND e.type = 'lane_deviation') as lane_count,
            (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id AND e.type = 'overspeed') as overspeed_count
            FROM vehicles v WHERE v.owner_id = ?
            ORDER BY v.created_at DESC""", (user['user_id'],))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify(rows)

    # POST — add vehicle
    data = request.get_json(force=True)
    vid = data.get('vehicle_id', '').strip()
    if not vid:
        conn.close()
        return jsonify({'error': 'vehicle_id is required'}), 400
    try:
        cur.execute("""INSERT INTO vehicles (vehicle_id, owner_id, label, speed_limit_kmh, notes, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (vid, user['user_id'], data.get('label',''), data.get('speed_limit_kmh'),
                     data.get('notes',''), datetime.now().isoformat()))
        conn.commit()
        new_id = cur.lastrowid
        conn.close()
        return jsonify({'ok': True, 'id': new_id})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Vehicle ID already exists'}), 400

@app.route('/api/user/vehicles/<int:id>', methods=['PUT', 'DELETE'])
@require_auth
def user_vehicle_detail(id):
    user = get_current_user()
    conn = get_db(); cur = conn.cursor()

    # Verify ownership
    cur.execute("SELECT * FROM vehicles WHERE id=? AND owner_id=?", (id, user['user_id']))
    vehicle = cur.fetchone()
    if not vehicle:
        conn.close()
        return jsonify({'error': 'Vehicle not found'}), 404

    if request.method == 'PUT':
        data = request.get_json(force=True)
        cur.execute("""UPDATE vehicles SET label=?, speed_limit_kmh=?, lane_sensitivity_px=?,
                       notes=?, updated_at=? WHERE id=?""",
                    (data.get('label', vehicle['label']),
                     data.get('speed_limit_kmh', vehicle['speed_limit_kmh']),
                     data.get('lane_sensitivity_px', vehicle['lane_sensitivity_px']),
                     data.get('notes', vehicle['notes']),
                     datetime.now().isoformat(), id))
        conn.commit(); conn.close()
        return jsonify({'ok': True})

    # DELETE
    cur.execute("DELETE FROM events_v2 WHERE vehicle_id=?", (vehicle['vehicle_id'],))
    cur.execute("DELETE FROM vehicles WHERE id=?", (id,))
    conn.commit(); conn.close()
    return jsonify({'ok': True})

@app.route('/api/user/vehicles/<int:id>/events')
@require_auth
def user_vehicle_events(id):
    user = get_current_user()
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT vehicle_id FROM vehicles WHERE id=? AND owner_id=?", (id, user['user_id']))
    vehicle = cur.fetchone()
    if not vehicle:
        conn.close()
        return jsonify({'error': 'Vehicle not found'}), 404
    cur.execute("SELECT * FROM events_v2 WHERE vehicle_id=? ORDER BY ts DESC", (vehicle['vehicle_id'],))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)

# ── Admin Endpoints ───────────────────────────────────────────────────────────

@app.route('/api/summary')
def global_summary():
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users WHERE role='user'")
    user_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM vehicles")
    vehicle_count = cur.fetchone()[0]
    lane_count = 0; overspeed_count = 0
    try:
        cur.execute("SELECT COUNT(*) FROM events_v2 WHERE type='lane_deviation'")
        lane_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM events_v2 WHERE type='overspeed'")
        overspeed_count = cur.fetchone()[0]
    except: pass
    conn.close()
    return jsonify({'user_count': user_count, 'vehicle_count': vehicle_count,
                    'lane_count': lane_count, 'overspeed_count': overspeed_count})

@app.route('/api/admin/users', methods=['GET', 'POST'])
@require_admin
def admin_users():
    conn = get_db(); cur = conn.cursor()
    if request.method == 'GET':
        cur.execute("""SELECT u.*, (SELECT COUNT(*) FROM vehicles v WHERE v.owner_id = u.id) as vehicle_count
                       FROM users u ORDER BY u.created_at DESC""")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify(rows)
    # POST
    data = request.get_json(force=True)
    ph = hashlib.sha256((data.get('password','') or '').encode()).hexdigest()
    try:
        cur.execute("INSERT INTO users (email,name,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                    (data.get('email'), data.get('name'), ph, data.get('role','user'), datetime.now().isoformat()))
        conn.commit(); uid = cur.lastrowid; conn.close()
        return jsonify({'ok': True, 'id': uid})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Email already exists'}), 400

@app.route('/api/admin/users/<int:id>', methods=['PUT', 'DELETE'])
@require_admin
def admin_user_detail(id):
    conn = get_db(); cur = conn.cursor()
    if request.method == 'PUT':
        data = request.get_json(force=True)
        cur.execute("UPDATE users SET name=?, email=?, role=?, updated_at=? WHERE id=?",
                    (data.get('name'), data.get('email'), data.get('role','user'), datetime.now().isoformat(), id))
        conn.commit(); conn.close()
        return jsonify({'ok': True})
    # DELETE
    cur.execute("DELETE FROM vehicles WHERE owner_id=?", (id,))
    cur.execute("DELETE FROM users WHERE id=?", (id,))
    conn.commit(); conn.close()
    return jsonify({'ok': True})

@app.route('/api/admin/vehicles', methods=['GET', 'POST'])
@require_admin
def admin_vehicles():
    conn = get_db(); cur = conn.cursor()
    if request.method == 'GET':
        cur.execute("""SELECT v.*, u.name as owner_name, u.email as owner_email,
            (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id) as total_events,
            (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id AND e.type='lane_deviation') as lane_count,
            (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id AND e.type='overspeed') as overspeed_count
            FROM vehicles v LEFT JOIN users u ON v.owner_id = u.id
            ORDER BY v.created_at DESC""")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify(rows)
    # POST — add new vehicle
    data = request.get_json(force=True)
    try:
        cur.execute(
            "INSERT INTO vehicles (vehicle_id,owner_id,label,speed_limit_kmh,kmh_per_pixel,notes,created_at) VALUES (?,?,?,?,?,?,?)",
            (data.get('vehicle_id'), data.get('owner_id'), data.get('label'),
             data.get('speed_limit_kmh'), data.get('kmh_per_pixel'), data.get('notes'), datetime.now().isoformat()))
        conn.commit(); vid = cur.lastrowid; conn.close()
        return jsonify({'ok': True, 'id': vid})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Vehicle ID already exists'}), 400

@app.route('/api/admin/vehicles/<int:id>', methods=['GET', 'PUT', 'DELETE'])
@require_admin
def admin_vehicle_detail(id):
    conn = get_db(); cur = conn.cursor()
    if request.method == 'GET':
        cur.execute("""SELECT v.*, u.name as owner_name, u.email as owner_email,
            (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id) as total_events,
            (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id AND e.type='lane_deviation') as lane_count,
            (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id AND e.type='overspeed') as overspeed_count
            FROM vehicles v LEFT JOIN users u ON v.owner_id = u.id WHERE v.id=?""", (id,))
        row = cur.fetchone(); conn.close()
        if not row:
            return jsonify({'error': 'Vehicle not found'}), 404
        return jsonify(dict(row))
    if request.method == 'PUT':
        data = request.get_json(force=True)
        cur.execute("UPDATE vehicles SET label=?, speed_limit_kmh=?, kmh_per_pixel=?, owner_id=?, notes=?, updated_at=? WHERE id=?",
                    (data.get('label'), data.get('speed_limit_kmh'), data.get('kmh_per_pixel'),
                     data.get('owner_id'), data.get('notes'), datetime.now().isoformat(), id))
        conn.commit(); conn.close()
        return jsonify({'ok': True})
    # DELETE
    cur.execute("SELECT vehicle_id FROM vehicles WHERE id=?", (id,))
    row = cur.fetchone()
    if row:
        cur.execute("DELETE FROM events_v2 WHERE vehicle_id=?", (row['vehicle_id'],))
    cur.execute("DELETE FROM vehicles WHERE id=?", (id,))
    conn.commit(); conn.close()
    return jsonify({'ok': True})

@app.route('/api/admin/vehicles/<int:id>/events')
@require_admin
def admin_vehicle_events(id):
    """Get all violation events for a vehicle (admin — no ownership check)."""
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT vehicle_id FROM vehicles WHERE id=?", (id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Vehicle not found'}), 404
    vid = row['vehicle_id']
    cur.execute("SELECT * FROM events_v2 WHERE vehicle_id=? ORDER BY ts DESC", (vid,))
    events = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(events)

# ── Calibration endpoints ────────────────────────────────────────────────────
@app.route('/api/admin/calibration/upload', methods=['POST'])
@require_admin
def admin_calibration_upload():
    """Accept two videos (video_30 / video_60), run auto_calibrate, return result."""
    if 'video_30' not in request.files or 'video_60' not in request.files:
        return jsonify({'error': 'Both video_30 and video_60 are required'}), 400

    from auto_calibrate import calibrate

    tmpdir = tempfile.mkdtemp()
    try:
        path_30   = os.path.join(tmpdir, 'v30.mp4')
        path_60   = os.path.join(tmpdir, 'v60.mp4')
        calib_out = os.path.join(tmpdir, 'calib.json')

        request.files['video_30'].save(path_30)
        request.files['video_60'].save(path_60)

        ok = calibrate(path_30, path_60, output_file=calib_out)
        if not ok:
            return jsonify({'error': 'Calibration failed — could not extract enough optical flow samples. Check video quality and make sure the vehicle was moving at a steady speed.'}), 422

        with open(calib_out) as f:
            result = json.load(f)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route('/api/admin/vehicles/<int:id>/calibrate', methods=['POST'])
@require_admin
def admin_save_calibration(id):
    """Save kmh_per_pixel calibration value to a vehicle record."""
    data = request.get_json(force=True)
    kpp = data.get('kmh_per_pixel')
    if kpp is None:
        return jsonify({'error': 'kmh_per_pixel required'}), 400
    conn = get_db(); cur = conn.cursor()
    cur.execute("UPDATE vehicles SET kmh_per_pixel=?, updated_at=? WHERE id=?",
                (float(kpp), datetime.now().isoformat(), id))
    if cur.rowcount == 0:
        conn.close()
        return jsonify({'error': 'Vehicle not found'}), 404
    conn.commit(); conn.close()
    return jsonify({'ok': True})


# ── Static files ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(os.path.join(PROJECT_ROOT, 'frontend'), 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'frontend'), path)

@app.route('/evidence/<path:filename>')
def serve_evidence(filename):
    """Serve evidence: redirect to Cloudinary URL if available, else local."""
    from flask import redirect
    cloud_url = get_evidence_url(filename)
    if cloud_url:
        return redirect(cloud_url)
    return send_from_directory(EVIDENCE_DIR, filename)




# Sensor CSV Processing Endpoint
import csv
from behavior.sensor_processing import KalmanFilter, extract_features
from behavior.inference import BehaviorInferenceService

# Lazy load model
behavior_model = None


def _as_builtin_json(value):
    if isinstance(value, dict):
        return {key: _as_builtin_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin_json(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _parse_behavior_coordinates():
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    if lat in (None, "") or lon in (None, ""):
        return None, None
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None


def _extract_behavior_score(details):
    probs = details.get("probs") or []
    if probs:
        try:
            return float(max(probs))
        except Exception:
            pass
    return 1.0 if details.get("override") else 0.0


def _insert_behavior_events(events):
    if not events:
        return
    conn = get_db()
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO driver_behavior (ts, behavior, score, latitude, longitude, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        events,
    )
    conn.commit()
    conn.close()


@app.route("/api/behavior/history", methods=["GET"])
def behavior_history():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ts, behavior, score, latitude, longitude, metadata
            FROM driver_behavior
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY ts DESC
            LIMIT 1000
            """
        )
        rows = cur.fetchall()
        conn.close()

        events = []
        for row in rows:
            metadata = {}
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                except Exception:
                    metadata = {"raw": row["metadata"]}

            events.append({
                "id": row["id"],
                "timestamp": row["ts"],
                "behavior": row["behavior"],
                "score": float(row["score"] or 0.0),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "meta": metadata,
            })

        return jsonify({"ok": True, "events": events})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "events": []}), 500

@app.route("/api/behavior/upload_csv", methods=["POST"])
def upload_sensor_csv():
    global behavior_model
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Must be a CSV file"}), 400

    try:
        # 1. Parse CSV
        stream = file.stream.read().decode("utf-8")
        reader = csv.DictReader(stream.splitlines())
        
        raw_data = []
        for row in reader:
            raw_data.append({
                "accelX": float(row.get("accelX", 0)),
                "accelY": float(row.get("accelY", 0)),
                "accelZ": float(row.get("accelZ", 0)),
                "gyroX": float(row.get("gyroX", 0)),
                "gyroY": float(row.get("gyroY", 0)),
                "gyroZ": float(row.get("gyroZ", 0)),
                "speed": float(row.get("speed", 0))
            })

        if len(raw_data) < 30:
            return jsonify({"error": f"Need at least 30 rows (1.5s), got {len(raw_data)}"}), 400

        # Process only the first 30 rows for 1 inference
        buffer_data = raw_data[:30]

        # 2. Kalman Filtering (simulate processData per axis)
        filters = {
            'ax': KalmanFilter(0.1, 0.1),
            'ay': KalmanFilter(0.5, 0.1),
            'az': KalmanFilter(0.1, 0.1),
            'gx': KalmanFilter(0.1, 0.1),
            'gy': KalmanFilter(0.1, 0.1),
            'gz': KalmanFilter(0.5, 0.1)
        }

        smoothed_data = []
        for row in buffer_data:
            smoothed_data.append({
                "accelX": filters['ax'].filter(row["accelX"]),
                "accelY": filters['ay'].filter(row["accelY"]),
                "accelZ": filters['az'].filter(row["accelZ"]),
                "gyroX": filters['gx'].filter(row["gyroX"]),
                "gyroY": filters['gy'].filter(row["gyroY"]),
                "gyroZ": filters['gz'].filter(row["gyroZ"]),
                "speed": row["speed"]
            })

        # 3. Extract Features — use raw 210-feature extraction (model requires it)
        from behavior.sensor_processing import extract_features_raw
        features = extract_features_raw(smoothed_data)

        # 4. Inference (PKL + Physics limits)
        if behavior_model is None:
            model_dir = os.path.join(BACKEND_DIR, "models/behavior_model")
            behavior_model = BehaviorInferenceService(model_dir)
        
        prediction, details = behavior_model.run_inference(features)

        # 5. Log to Database
        latitude, longitude = _parse_behavior_coordinates()
        min_ax = min(row["accelX"] for row in buffer_data)
        max_ay = max(abs(row["accelY"]) for row in buffer_data)
        max_speed = max(row["speed"] for row in buffer_data)
        metadata = json.dumps(_as_builtin_json({
            "trigger": details.get("trigger"),
            "override": details.get("override"),
            "speed": max_speed,
            "max_ax": min_ax,
            "max_ay": max_ay,
        }))
        _insert_behavior_events([(
            datetime.now().isoformat(),
            prediction,
            _extract_behavior_score(details),
            latitude,
            longitude,
            metadata,
        )])

        return jsonify({
            "ok": True,
            "prediction": prediction,
            "details": details,
            "features_extracted": len(features)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Continuous Time-Series Analysis Endpoint ──────────────────────────────────
import pandas as pd

@app.route("/api/behavior/upload_continuous", methods=["POST"])
def upload_continuous_csv():
    """
    Accepts a large continuous CSV file. Applies:
      1. Kalman Filter smoothing on accel_x, accel_y
      2. Sliding Window (N=30, S=5) feature extraction
      3. ONNX model + Physics Soft Fusion per window
    Returns a JSON array of per-window results for charting.
    """
    global behavior_model

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Must be a CSV file"}), 400

    try:
        # ── 1. Parse CSV with pandas ──────────────────────────────────────
        df = pd.read_csv(file.stream)

        # Normalise column names (support both naming conventions)
        rename_map = {
            'accel_x': 'accelX', 'accel_y': 'accelY', 'accel_z': 'accelZ',
            'gyro_x': 'gyroX', 'gyro_y': 'gyroY', 'gyro_z': 'gyroZ',
            'event_type': 'actual_label'
        }
        df.rename(columns=rename_map, inplace=True)

        # Add speed=0 if column missing (IMU-only data)
        if 'speed' not in df.columns:
            df['speed'] = 0.0

        # Add actual_label if missing
        if 'actual_label' not in df.columns:
            df['actual_label'] = 'Unknown'

        required = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']
        for col in required:
            if col not in df.columns:
                return jsonify({"error": f"Missing required column: {col}"}), 400

        total_rows = len(df)
        if total_rows < 30:
            return jsonify({"error": f"Need at least 30 rows, got {total_rows}"}), 400

        latitude, longitude = _parse_behavior_coordinates()

        # ── 2. Kalman Filter smoothing ────────────────────────────────────
        kf_ax = KalmanFilter(R=0.1, Q=0.1)
        kf_ay = KalmanFilter(R=0.5, Q=0.1)

        smoothed_ax = []
        smoothed_ay = []
        for i in range(total_rows):
            smoothed_ax.append(kf_ax.filter(float(df.iloc[i]['accelX'])))
            smoothed_ay.append(kf_ay.filter(float(df.iloc[i]['accelY'])))

        df['accelX_smooth'] = smoothed_ax
        df['accelY_smooth'] = smoothed_ay

        # ── 3. Sliding Window (N=30, S=5) ────────────────────────────────
        WINDOW_SIZE = 30
        STEP_SIZE = 5
        ALPHA = 0.1  # Physics soft fusion weight

        # Lazy-load model
        if behavior_model is None:
            model_dir = os.path.join(BACKEND_DIR, "models/behavior_model")
            behavior_model = BehaviorInferenceService(model_dir)

        results = []
        events_to_log = []
        window_idx = 0

        for start in range(0, total_rows - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            window = df.iloc[start:end]

            # Build buffer_data for feature extraction (use smoothed accel)
            buffer_data = []
            for _, row in window.iterrows():
                buffer_data.append({
                    "accelX": float(row['accelX_smooth']),
                    "accelY": float(row['accelY_smooth']),
                    "accelZ": float(row['accelZ']),
                    "gyroX": float(row['gyroX']),
                    "gyroY": float(row['gyroY']),
                    "gyroZ": float(row['gyroZ']),
                    "speed": float(row['speed'])
                })

            # Extract 210-feature raw vector (model requires it)
            from behavior.sensor_processing import extract_features_raw
            features = extract_features_raw(buffer_data)

            # ── Physics Soft Fusion (α boost) ─────────────────────────────
            # Before running the model, check physics thresholds
            raw_ax = [float(r['accelX']) for r in buffer_data]
            raw_ay = [float(r['accelY']) for r in buffer_data]
            min_ax = min(raw_ax)
            max_ay = max(abs(a) for a in raw_ay)

            # Run inference
            prediction, details = behavior_model.run_inference(features)

            # Apply soft fusion α-boost to probabilities
            if 'probs' in details and len(details['probs']) == 5:
                probs = list(details['probs'])
                # If min accelX < -3.0, boost Harsh_Brake probability
                if min_ax < -3.0:
                    probs[0] = min(1.0, probs[0] + ALPHA)
                    norm = sum(probs)
                    probs = [p / norm for p in probs]
                    details['probs'] = probs
                    details['soft_fusion'] = f'Brake boost (min_ax={min_ax:.2f})'
                # If max|accelY| > 3.0, boost Sharp_Turn probability
                if max_ay > 3.0:
                    probs[2] = min(1.0, probs[2] + ALPHA)
                    norm = sum(probs)
                    probs = [p / norm for p in probs]
                    details['probs'] = probs
                    details['soft_fusion'] = f'Turn boost (max_ay={max_ay:.2f})'

                # Re-determine prediction from boosted probs
                from behavior.inference import LABELS
                max_idx = int(np.argmax(probs))
                prediction = LABELS[max_idx]

            # Get the timestamp and actual label for this window
            ts_val = str(window.iloc[0].get('timestamp', f'Window_{window_idx}'))
            actual_label = str(window.iloc[15].get('actual_label', 'Unknown'))  # Mid-window label

            # Collect raw accelX values for the chart
            raw_accel_x = window['accelX'].tolist()

            results.append({
                "window_id": window_idx,
                "timestamp": ts_val,
                "start_idx": int(start),
                "end_idx": int(end),
                "predicted_label": prediction,
                "actual_label": actual_label,
                "trigger": details.get("override", details.get("trigger", "AI")),
                "probs": details.get("probs", []),
                "soft_fusion": details.get("soft_fusion", None),
                "raw_accelX": raw_accel_x
            })

            if prediction != 'Normal_Driving':
                events_to_log.append((
                    ts_val,
                    prediction,
                    _extract_behavior_score(details),
                    latitude,
                    longitude,
                    json.dumps(_as_builtin_json({
                        "trigger": details.get("trigger"),
                        "override": details.get("override"),
                        "soft_fusion": details.get("soft_fusion"),
                        "window_id": window_idx,
                        "start_idx": int(start),
                        "end_idx": int(end),
                        "speed": float(window['speed'].max()),
                        "max_ax": float(min(window['accelX'].tolist())),
                        "max_ay": float(max(abs(v) for v in window['accelY'].tolist())),
                    })),
                ))

            window_idx += 1

            # Safety: cap at 6000 windows for large files
            if window_idx >= 6000:
                break

        # ── 4. Build full accelX trace for the chart ──────────────────────
        # Subsample if extremely large
        max_chart_points = 15000
        step = max(1, total_rows // max_chart_points)
        chart_accelX = df['accelX'].iloc[::step].tolist()
        chart_accelX_smooth = df['accelX_smooth'].iloc[::step].tolist()
        chart_timestamps = df['timestamp'].iloc[::step].tolist() if 'timestamp' in df.columns else list(range(0, total_rows, step))

        _insert_behavior_events(events_to_log)

        return jsonify({
            "ok": True,
            "total_rows": total_rows,
            "total_windows": window_idx,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "windows": results,
            "chart": {
                "timestamps": chart_timestamps,
                "accelX_raw": chart_accelX,
                "accelX_smooth": chart_accelX_smooth
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Use threaded=True so stream + Flask routes run concurrently
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
