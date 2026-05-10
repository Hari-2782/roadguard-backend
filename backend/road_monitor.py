import cv2
import numpy as np
import sqlite3
import json
import sys
import warnings
import os
import time
from datetime import datetime

# RankWarning import compatible with numpy versions
try:
    from numpy.polynomial.polyutils import RankWarning
except Exception:
    try:
        from numpy import RankWarning  # older numpy
    except Exception:
        RankWarning = Warning  # fallback

# Optional TTS (don’t crash if not installed)
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

from centerline import detect_lane_lines, draw_lane_lines
from sign_detector import YOLOSignDetector, SignState, SPEED_VALUE, JUNCTION_CLASSES, JUNCTION_SPEECH

# ── Absolute root so paths work whether called directly or imported ──────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

# Try to use DL lane detector if model exists
USE_DL_LANE_DETECTOR = False
try:
    from dl_lane_detector import DLLaneDetector
    _lane_pth = os.path.join(_ROOT, "models", "lane_model", "lane_detector.pth")
    if os.path.exists(_lane_pth):
        dl_lane_detector = DLLaneDetector(_lane_pth)
        USE_DL_LANE_DETECTOR = True
        print("[INFO] Using Deep Learning lane detector")
    else:
        print("[INFO] DL lane model not found, using traditional CV detection")
except Exception as e:
    print(f"[INFO] DL lane detector not available: {e}")

warnings.simplefilter("ignore", RankWarning)

# =========================
# ---- Global config  -----
# =========================

# Evidence directory (must match api.py)
EVIDENCE_DIR = os.path.join(_ROOT, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

DB_PATH = os.path.join(_ROOT, "new.db")

YOLO_WEIGHTS = os.path.join(_ROOT, "runs", "detect", "runs", "detect",
                             "speed_junction_v1", "weights", "best.pt")
YOLO_CONF = 0.35  # Focused model is more precise, can use moderate threshold
YOLO_IMG_SIZE = 640

DEFAULT_SPEED_LIMIT_KMH = 50

OVERSPEED_COOLDOWN_FRAMES = 30
OVERSPEED_TOLERANCE_KMH = 8 # Allow 8 km/h grace period before screaming

# Default lane sensitivity fallback (can be overwritten from DB per vehicle)
DEFAULT_LANE_DEVIATION_BAND_PX = 40
LANE_COOLDOWN_FRAMES = 300  # ~5 seconds at 60fps - capture only ONE image per violation

# Calibration fallback
FALLBACK_KMH_PER_PIXEL = 1.0
DEFAULT_KMH_PER_PIXEL = 1.0
DEFAULT_SPEED_OFFSET = 0.0
CALIB_FILE = os.path.join(_ROOT, "calibration.json")

CALIB_SPEED1 = 30
CALIB_SPEED2 = 60
CALIB_MIN_SAMPLES = 50  # increased for stability

# =========================
# ---- Optical Flow Config (BALANCED - for accuracy + stability) -----
# =========================
# ROI for optical flow: Track road area (balanced for accuracy)
FLOW_ROI_Y_START = 0.50   # Start at 50% down
FLOW_ROI_Y_END = 0.90     # End at 90%
FLOW_ROI_X_START = 0.20   # Wider for more road coverage
FLOW_ROI_X_END = 0.80     # Wider for more road coverage

# Use median for robustness (less affected by outliers like passing cars)
FLOW_USE_MEDIAN = True

# Perspective weighting: DISABLED to match calibration tool logic
FLOW_PERSPECTIVE_WEIGHT = False

# Speed smoothing (EMA alpha: VERY LOW for maximum stability)
# Lower = more stable but slower to respond
SPEED_SMOOTHING_ALPHA = 0.20  # Higher = faster response but more jitter

# Minimum flow threshold to consider as motion (filters noise when stopped)
FLOW_MIN_THRESHOLD = 0.5
# Junction filters (balanced: filter noise, allow real signs)
JUNCTION_MIN_CONF = 0.70            # Raised to 0.70 - Research Grade precision
JUNCTION_MIN_AREA_RATIO = 0.0001     # minimum area (Lowered for custom model)
JUNCTION_MAX_AREA_RATIO = 0.08      # allow larger signs
JUNCTION_ASPECT_MIN = 0.50          # looser aspect ratio
JUNCTION_ASPECT_MAX = 2.00          # looser aspect ratio
JUNCTION_MAX_Y1_RATIO = 0.60        # must be in upper 60% of frame
JUNCTION_HORIZONTAL_MARGIN = 0.10   # REDUCED - allow more positions (left 10% or right 90%)

# Speed sign filters - focused model (more precise, simpler filters)
SPEED_SIGN_MIN_CONF = 0.35          # Focused model has higher confidence for true positives
SPEED_SIGN_MIN_AREA_RATIO = 0.0001   # minimum area
SPEED_SIGN_MAX_AREA_RATIO = 0.25    # allow close-up signs
SPEED_SIGN_MAX_Y1_RATIO = 0.85      # signs must be in upper 85% of frame
SPEED_SIGN_HORIZONTAL_MARGIN = 0.05 # minimal horizontal filter


# =========================
# ---- DB + logging -------
# =========================

def init_db():
    """
    Create/migrate tables compatible with both road_monitor and api_server.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Events table — minimal schema; api_server adds extra columns via ALTER
    cursor.execute("""
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
    # Backwards-compat: add columns if they were created without them
    for col in ("image_path TEXT", "risk_level TEXT"):
        try:
            cursor.execute(f"ALTER TABLE events_v2 ADD COLUMN {col}")
        except Exception:
            pass

    # Vehicles table — only create if not present; api_server owns its schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT UNIQUE,
            owner_id INTEGER,
            label TEXT,
            speed_limit_kmh REAL,
            lane_sensitivity_px REAL,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn, cursor


def log_event(cursor, conn, event_type, vehicle_id, value, details, image_path=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO events_v2(ts, type, value, details, vehicle_id, image_path) VALUES (?, ?, ?, ?, ?, ?)",
        (ts, event_type, value, details, vehicle_id, image_path),
    )
    conn.commit()
    print(f"[LOG] {ts} | {event_type} | value={value} | img={image_path} | {details}")


def load_vehicle_config(conn, vehicle_id: str):
    """
    Reads per-vehicle configuration from DB.
    Returns dict: {speed_limit_kmh, lane_sensitivity_px, label, notes}
    """
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT label, speed_limit_kmh, lane_sensitivity_px, notes FROM vehicles WHERE vehicle_id=?",
            (vehicle_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"label": None, "speed_limit_kmh": None, "lane_sensitivity_px": None, "notes": None}
        label, speed_limit_kmh, lane_sensitivity_px, notes = row
        return {
            "label": label,
            "speed_limit_kmh": speed_limit_kmh,
            "lane_sensitivity_px": lane_sensitivity_px,
            "notes": notes,
        }
    except Exception:
        return {"label": None, "speed_limit_kmh": None, "lane_sensitivity_px": None, "notes": None}


# =========================
# ---- TTS init -----------
# =========================

def init_tts():
    if pyttsx3 is None:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        return engine
    except Exception:
        return None


# =========================
# ---- Calibration logic --
# =========================

def load_calibration(calib_path=CALIB_FILE):
    try:
        with open(calib_path, "r") as f:
            data = json.load(f)
        kmh_per_pixel = float(data.get("kmh_per_pixel", DEFAULT_KMH_PER_PIXEL))
        speed_offset = float(data.get("offset", DEFAULT_SPEED_OFFSET))
        
        if kmh_per_pixel <= 0:
            # If invalid factor, warn but allow offset? No, fallback.
            raise ValueError("Invalid kmh_per_pixel")
            
        print(f"[INFO] Loaded calibration: kmh_per_pixel = {kmh_per_pixel:.6f}, offset = {speed_offset:.6f}")
        return kmh_per_pixel, speed_offset
    except Exception as e:
        print(f"[WARN] No valid calibration file ({e}). Using fallback: {FALLBACK_KMH_PER_PIXEL}")
        return FALLBACK_KMH_PER_PIXEL, DEFAULT_SPEED_OFFSET



def save_calibration(kmh_per_pixel, flow1, flow2, fps, offset=DEFAULT_SPEED_OFFSET, path=CALIB_FILE):
    """Save calibration with FPS info for reproducibility."""
    data = {
        "kmh_per_pixel": kmh_per_pixel,
        "offset": offset,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_fps": fps,
        "calib_points": {
            "speed1_kmh": CALIB_SPEED1,
            "flow1_px_per_frame": flow1,
            "speed2_kmh": CALIB_SPEED2,
            "flow2_px_per_frame": flow2,
        },
        "flow_config": {
            "roi": [FLOW_ROI_Y_START, FLOW_ROI_Y_END, FLOW_ROI_X_START, FLOW_ROI_X_END],
            "use_median": FLOW_USE_MEDIAN,
            "perspective_weight": FLOW_PERSPECTIVE_WEIGHT,
        }
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Calibration saved to {path}")
    print(f"[INFO] kmh_per_pixel = {kmh_per_pixel:.6f}, FPS = {fps:.2f}")


def get_roi_mask(h, w):
    """Create a mask for the road ROI (bottom-center of frame)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    y1 = int(h * FLOW_ROI_Y_START)
    y2 = int(h * FLOW_ROI_Y_END)
    x1 = int(w * FLOW_ROI_X_START)
    x2 = int(w * FLOW_ROI_X_END)
    mask[y1:y2, x1:x2] = 255
    return mask


def compute_optical_flow_speed(old_gray, frame_gray, p0, lk_params, h, w):
    """
    Compute optical flow with improvements:
    1. ROI restriction (only road surface)
    2. Median instead of mean (robust to outliers)
    3. Perspective weighting (near = more weight)
    
    Returns: (flow_value, new_p0)
    """
    if p0 is None or len(p0) < 10:
        # Re-detect points in ROI only
        roi_mask = get_roi_mask(h, w)
        p0 = cv2.goodFeaturesToTrack(
            old_gray, maxCorners=300, qualityLevel=0.2, 
            minDistance=7, blockSize=7, mask=roi_mask
        )
        if p0 is None or len(p0) < 10:
            return 0.0, p0
    
    # Track points
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is None:
        return 0.0, None
    
    # Filter good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    if len(good_new) < 5:
        return 0.0, None
    
    # Filter points that are in ROI
    y1_roi = int(h * FLOW_ROI_Y_START)
    y2_roi = int(h * FLOW_ROI_Y_END)
    x1_roi = int(w * FLOW_ROI_X_START)
    x2_roi = int(w * FLOW_ROI_X_END)
    
    roi_mask_pts = (
        (good_old[:, 0] >= x1_roi) & (good_old[:, 0] <= x2_roi) &
        (good_old[:, 1] >= y1_roi) & (good_old[:, 1] <= y2_roi)
    )
    
    good_new = good_new[roi_mask_pts]
    good_old = good_old[roi_mask_pts]
    
    if len(good_new) < 5:
        return 0.0, None
    
    # Compute displacements
    displacements = np.linalg.norm(good_new - good_old, axis=1)
    
    if FLOW_PERSPECTIVE_WEIGHT:
        # Weight by Y position: points closer to bottom (larger Y) get more weight
        # Normalize Y to [0, 1] within ROI, then use as weight
        y_positions = good_old[:, 1]
        weights = (y_positions - y1_roi) / max(1, y2_roi - y1_roi)
        weights = np.clip(weights, 0.3, 1.0)  # minimum weight 0.3
        
        if FLOW_USE_MEDIAN:
            # Weighted median approximation: repeat values by weight
            weighted_displacements = []
            for d, w_val in zip(displacements, weights):
                # Repeat based on weight (scaled to int)
                count = max(1, int(w_val * 10))
                weighted_displacements.extend([d] * count)
            flow_value = float(np.median(weighted_displacements))
        else:
            # Weighted mean
            flow_value = float(np.average(displacements, weights=weights))
    else:
        if FLOW_USE_MEDIAN:
            flow_value = float(np.median(displacements))
        else:
            flow_value = float(np.mean(displacements))
    
    # Apply minimum threshold (noise filter when stopped)
    if flow_value < FLOW_MIN_THRESHOLD:
        flow_value = 0.0
    
    # Re-detect points for next frame (within ROI)
    roi_mask = get_roi_mask(h, w)
    new_p0 = cv2.goodFeaturesToTrack(
        frame_gray, maxCorners=300, qualityLevel=0.2,
        minDistance=7, blockSize=7, mask=roi_mask
    )
    
    return flow_value, new_p0


def check_road_visible(frame, h, w):
    """
    Check if the road is visible in the ROI.
    Returns False if the ROI is mostly sky (blue/white) or trees (green).
    This prevents optical flow from tracking non-road features.
    """
    # Get ROI
    y1 = int(h * FLOW_ROI_Y_START)
    y2 = int(h * FLOW_ROI_Y_END)
    x1 = int(w * FLOW_ROI_X_START)
    x2 = int(w * FLOW_ROI_X_END)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Detect GREEN (trees/foliage) - Hue 35-85
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = cv2.countNonZero(mask_green) / (roi.shape[0] * roi.shape[1])
    
    # Detect SKY (light blue/white/gray) - Low saturation, high value
    lower_sky = np.array([0, 0, 150])
    upper_sky = np.array([180, 60, 255])
    mask_sky = cv2.inRange(hsv, lower_sky, upper_sky)
    sky_ratio = cv2.countNonZero(mask_sky) / (roi.shape[0] * roi.shape[1])
    
    # If more than 60% is green OR more than 70% is sky, road is not visible
    # (Relaxed thresholds to work in more road conditions)
    if green_ratio > 0.60:
        return False
    if sky_ratio > 0.70:
        return False
    
    return True


def run_calibration(video_source):
    """
    Improved calibration mode:
    - Uses ROI-restricted optical flow (road surface only)
    - Shows real-time flow statistics
    - Captures FPS for reproducibility
    - Uses median for robustness
    """
    print("=== IMPROVED CALIBRATION MODE ===")
    print(f"Press '1' to START capturing at ~{CALIB_SPEED1} km/h")
    print(f"Press '2' to START capturing at ~{CALIB_SPEED2} km/h")
    print(f"Press 's' to STOP capturing")
    print(f"Need at least {CALIB_MIN_SAMPLES} samples for each speed.")
    print("Press 'c' to compute calibration when ready.")
    print("Press 'q' to quit.\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {source}")
        return

    # Video Writer
    out_file = "demo_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
    print(f"[INFO] Recording demo to {out_file}...")

    sign_detector = YOLOSignDetector(YOLO_WEIGHTS, conf=YOLO_CONF, imgsz=YOLO_IMG_SIZE)
    if fps <= 0:
        fps = 30.0  # fallback
    print(f"[INFO] Video FPS: {fps:.2f}")

    ret, old_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame.")
        cap.release()
        return

    h, w = old_frame.shape[:2]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize tracking points in ROI
    roi_mask = get_roi_mask(h, w)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=300, qualityLevel=0.2, 
                                  minDistance=7, blockSize=7, mask=roi_mask)

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    flows_1, flows_2 = [], []
    capture_mode = None  # None, 1, or 2

    # Simple debounce
    last_key = None
    last_key_time = 0.0
    debounce_sec = 0.30

    # Real-time flow history for display
    flow_history = []
    FLOW_HISTORY_LEN = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video for calibration
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use improved optical flow function
        flow_value, p0 = compute_optical_flow_speed(old_gray, frame_gray, p0, lk_params, h, w)
        old_gray = frame_gray.copy()

        # Track flow history
        flow_history.append(flow_value)
        if len(flow_history) > FLOW_HISTORY_LEN:
            flow_history.pop(0)
        
        avg_recent_flow = np.median(flow_history) if flow_history else 0.0

        # Capture if in mode
        if capture_mode == 1 and flow_value > FLOW_MIN_THRESHOLD:
            flows_1.append(flow_value)
        elif capture_mode == 2 and flow_value > FLOW_MIN_THRESHOLD:
            flows_2.append(flow_value)

        # Draw HUD
        hud = frame.copy()
        
        # Draw ROI rectangle
        y1_roi = int(h * FLOW_ROI_Y_START)
        y2_roi = int(h * FLOW_ROI_Y_END)
        x1_roi = int(w * FLOW_ROI_X_START)
        x2_roi = int(w * FLOW_ROI_X_END)
        cv2.rectangle(hud, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 255), 2)
        cv2.putText(hud, "ROI", (x1_roi + 5, y1_roi + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Title
        cv2.putText(hud, "CALIBRATION MODE (Improved)", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Speed 1 info
        flow1_avg = np.median(flows_1) if flows_1 else 0.0
        flow1_std = np.std(flows_1) if len(flows_1) > 1 else 0.0
        color1 = (0, 255, 0) if capture_mode == 1 else (200, 200, 200)
        cv2.putText(hud, f"[1] {CALIB_SPEED1} km/h: {len(flows_1)} samples, median={flow1_avg:.2f}, std={flow1_std:.2f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color1, 2)
        
        # Speed 2 info
        flow2_avg = np.median(flows_2) if flows_2 else 0.0
        flow2_std = np.std(flows_2) if len(flows_2) > 1 else 0.0
        color2 = (0, 255, 0) if capture_mode == 2 else (200, 200, 200)
        cv2.putText(hud, f"[2] {CALIB_SPEED2} km/h: {len(flows_2)} samples, median={flow2_avg:.2f}, std={flow2_std:.2f}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color2, 2)
        
        # Current flow
        cv2.putText(hud, f"Current flow: {flow_value:.2f} px/frame | Recent avg: {avg_recent_flow:.2f}", 
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        status_text = "CAPTURING" if capture_mode else "STOPPED"
        cv2.putText(hud, f"Status: {status_text} | Press 's' to stop, 'c' to compute, 'q' to quit", 
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Calibration", hud)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        # Debounce
        if key in (ord("1"), ord("2"), ord("s"), ord("c"), ord("q"), 27):
            if last_key == key and (now - last_key_time) < debounce_sec:
                key = 0
            else:
                last_key = key
                last_key_time = now

        if key == ord("1"):
            capture_mode = 1
            print(f"[INFO] Started capturing for {CALIB_SPEED1} km/h...")
        elif key == ord("2"):
            capture_mode = 2
            print(f"[INFO] Started capturing for {CALIB_SPEED2} km/h...")
        elif key == ord("s"):
            capture_mode = None
            print("[INFO] Stopped capturing.")
        elif key == ord("c"):
            # Try to compute calibration
            if len(flows_1) >= CALIB_MIN_SAMPLES and len(flows_2) >= CALIB_MIN_SAMPLES:
                break
            else:
                print(f"[WARN] Need at least {CALIB_MIN_SAMPLES} samples for each speed.")
                print(f"       Speed1: {len(flows_1)}, Speed2: {len(flows_2)}")
        elif key == ord("q") or key == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Calibration cancelled.")
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(flows_1) < CALIB_MIN_SAMPLES or len(flows_2) < CALIB_MIN_SAMPLES:
        print("[ERROR] Not enough samples collected.")
        return

    # Use median for robustness
    flow1 = float(np.median(flows_1))
    flow2 = float(np.median(flows_2))
    
    print(f"\n[RESULT] Flow at {CALIB_SPEED1} km/h: median = {flow1:.4f} px/frame (std = {np.std(flows_1):.4f})")
    print(f"[RESULT] Flow at {CALIB_SPEED2} km/h: median = {flow2:.4f} px/frame (std = {np.std(flows_2):.4f})")
    
    if abs(flow2 - flow1) < 0.5:
        print("[ERROR] Flow difference too small. Try driving at more distinct speeds.")
        return

    # Linear calibration: speed = flow * kmh_per_pixel + offset
    # For simplicity: kmh_per_pixel = (speed2 - speed1) / (flow2 - flow1)
    kmh_per_pixel = (CALIB_SPEED2 - CALIB_SPEED1) / (flow2 - flow1)
    
    # Compute offset for zero-intercept (optional improvement)
    # offset = speed1 - flow1 * kmh_per_pixel
    
    print(f"[RESULT] kmh_per_pixel = {kmh_per_pixel:.6f}")
    
    save_calibration(kmh_per_pixel, flow1, flow2, fps)


# =========================
# ---- Monitoring logic ---
# =========================

def is_valid_junction_box(d, frame_w, frame_h):
    """
    Reduces junction false positives using strict geometry + position constraints.
    Road signs in Sri Lanka appear:
    - Upper portion of frame (not on road surface)
    - Left or right side (not center of frame)
    - Roughly square aspect ratio
    """
    # Higher confidence for junctions
    if d["conf"] < JUNCTION_MIN_CONF:
        return False

    x1, y1, x2, y2 = d["xyxy"]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    x_center = (x1 + x2) / 2
    
    area_ratio = (bw * bh) / float(frame_w * frame_h)
    
    # --- HYBRID LOGIC: Classification Fallback ---
    # If the box is huge (> 50% of screen), treat it as a "Classification Event"
    # and BYPASS the geometry/position filters.
    if area_ratio > 0.50:
        return True
    # ---------------------------------------------

    aspect = bw / float(bh)

    # Size constraints
    if area_ratio < JUNCTION_MIN_AREA_RATIO:
        return False  # too small (noise)
    if area_ratio > JUNCTION_MAX_AREA_RATIO:
        return False  # too big (false detection)
    
    # Aspect ratio (signs are roughly square)
    if aspect < JUNCTION_ASPECT_MIN or aspect > JUNCTION_ASPECT_MAX:
        return False
    
    # Vertical position: must be in upper portion of frame
    if y1 > int(JUNCTION_MAX_Y1_RATIO * frame_h):
        return False
    
    # Horizontal position: signs appear on left or right side, not center
    # left_margin = JUNCTION_HORIZONTAL_MARGIN * frame_w
    # right_margin = (1 - JUNCTION_HORIZONTAL_MARGIN) * frame_w
    # if left_margin < x_center < right_margin:
    #     return False  # reject center detections (likely false positives)

    return True


def is_valid_speed_sign_box(d, frame_w, frame_h):
    """
    Simplified & Robust Validation.
    """
    # 1. Confidence Check
    if d["conf"] < SPEED_SIGN_MIN_CONF:
        return False

    x1, y1, x2, y2 = d["xyxy"]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    frame_area = max(1, float(frame_w * frame_h))
    area_ratio = (bw * bh) / frame_area
    
    # 2. Size Checks
    if area_ratio < SPEED_SIGN_MIN_AREA_RATIO:
        print(f"[DEBUG-GEO] {d['cls_name']} Too Small: {area_ratio:.5f}")
        return False
    if area_ratio > SPEED_SIGN_MAX_AREA_RATIO:
        print(f"[DEBUG-GEO] {d['cls_name']} Too Big: {area_ratio:.5f}")
        return False

    # 3. Position Check (Vertical)
    limit_y = int(SPEED_SIGN_MAX_Y1_RATIO * frame_h)
    if y1 > limit_y:
        print(f"[DEBUG-GEO] {d['cls_name']} Too Low: y1={y1} > {limit_y}")
        return False

    # 4. Aspect Ratio Check (STRICTER - Speed signs are circular/square)
    # Real speed signs: 0.8 - 1.25 aspect ratio
    # People/Trees/Poles: Aspect < 0.7 (tall)
    # Chevron boards: Aspect > 1.3 (wide/diamond)
    
    aspect = bw / float(bh)
    
    if aspect < 0.50:  # Relaxed from 0.80 to allow close/partial signs
        print(f"[DEBUG-GEO] {d['cls_name']} Too Tall: Aspect={aspect:.2f} (Likely Person/Pole)")
        return False
        
    if aspect > 1.25:
        print(f"[DEBUG-GEO] {d['cls_name']} Too Wide: Aspect={aspect:.2f} (Likely Chevron/Banner)")
        return False

    return True


def verify_speed_sign_circle(frame, d):
    """
    Lightweight check: Is the detected region CIRCULAR (real speed sign)
    or DIAMOND/SQUARE (false positive like a priority/warning sign)?
    
    Sri Lankan speed signs are ALWAYS round with a red border.
    Diamond-shaped priority signs and square warning signs should be rejected.
    
    Returns True if the shape is circular enough to be a speed sign.
    """
    x1, y1, x2, y2 = d["xyxy"]
    cls_name = d.get("cls_name", "Unknown")
    conf = d.get("conf", 0.5)
    
    h, w = frame.shape[:2]
    
    # Expand bounding box by 20% to include the red border
    # (YOLO may crop tightly on the white center when sign is close)
    bw = x2 - x1
    bh = y2 - y1
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
    
    # Too small to analyse reliably — let it through
    if roi_h < 25 or roi_w < 25:
        return True
    
    # === 1. CIRCULARITY CHECK ===
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Smooth noise for close-up signs
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circularity = 1.0  # Default: assume circular if no good contour
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        
        if perimeter > 0 and area > 100:
            # Circularity: 4π × area / perimeter²
            # Perfect circle = 1.0, Square = ~0.78, Diamond = ~0.5, Complex = <0.4
            circularity = (4 * 3.14159 * area) / (perimeter * perimeter)
    
    # High-confidence YOLO detections get a more lenient threshold
    circ_threshold = 0.25 if conf >= 0.85 else 0.40
    
    if circularity < circ_threshold:
        print(f"[DEBUG-CIRCLE] {cls_name} REJECTED: Circularity={circularity:.2f} conf={conf:.2f} (diamond/polygon shape)")
        return False
        
    # For high-confidence detections that pass circularity, trust YOLO
    if conf >= 0.85 and circularity >= 0.25:
        print(f"[DEBUG-CIRCLE] {cls_name} PASSED: Circularity={circularity:.2f} conf={conf:.2f} (high-conf)")
        return True
    
    # === RED BORDER CHECK (for lower-confidence detections) ===
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Strict Red detection: Saturation must be > 120 to ignore greyish-purple noise
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([165, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    
    red_ratio = cv2.countNonZero(mask_red) / (roi_h * roi_w)
    
    if circularity >= 0.55:
        if red_ratio < 0.005 and roi_w > 15:
            print(f"[DEBUG-CIRCLE] {cls_name} REJECTED: Round (Circ={circularity:.2f}) but NO RED ({red_ratio:.1%}) — likely false pos")
            return False
        print(f"[DEBUG-CIRCLE] {cls_name} PASSED: Circularity={circularity:.2f}")
        return True
    
    # === AMBIGUOUS ZONE (0.40-0.55): Use RED BORDER as tiebreaker ===
    # In the ambiguous zone, require at least 2% red
    if red_ratio < 0.02:
        print(f"[DEBUG-CIRCLE] {cls_name} REJECTED: Ambiguous shape (Circ={circularity:.2f}) + no red ({red_ratio:.1%})")
        return False
    
    print(f"[DEBUG-CIRCLE] {cls_name} PASSED: Circ={circularity:.2f} Red={red_ratio:.1%}")
    return True


def run_monitor(video_source, vehicle_id="unknown", headless=False, stats_callback=None):
    print(f"[INFO] Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source {video_source}")
        return

    # Video Writer (Disabled per user request)
    # out_file = "demo_output.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out_writer = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
    # print(f"[INFO] Recording demo to {out_file}...")

    # DB + TTS
    conn, cursor = init_db()
    engine = init_tts()

    # Load calibration
    kmh_per_pixel, speed_offset = load_calibration()

    # Load YOLO sign detector
    if not os.path.exists(YOLO_WEIGHTS):
        print(f"[ERROR] YOLO weights not found: {YOLO_WEIGHTS}")
        cap.release()
        conn.close()
        return

    sign_model = YOLOSignDetector(YOLO_WEIGHTS, conf=YOLO_CONF, imgsz=YOLO_IMG_SIZE)
    sign_state = SignState(
        speed_vote_window=12,       # balanced window
        speed_min_hits=3,           # STABILITY: Require 3 consecutive frames to confirm sign (prevents single-frame glitches)
        speed_hold_seconds=10.0,    # keep sign speed active for 10s
        junction_vote_window=10,    # balanced window
        junction_min_hits=2,        # Snappy junctions
        junction_cooldown_sec=6.0   # reasonable cooldown
    )

    # Load vehicle config from DB (admin/parent portal)
    cfg = load_vehicle_config(conn, vehicle_id)
    admin_speed_limit = cfg.get("speed_limit_kmh")
    lane_band_px = cfg.get("lane_sensitivity_px") or DEFAULT_LANE_DEVIATION_BAND_PX
    if admin_speed_limit is None:
        admin_speed_limit = DEFAULT_SPEED_LIMIT_KMH

    # Optical-flow init (improved with ROI)
    ret, old_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame.")
        cap.release()
        conn.close()
        return

    h, w = old_frame.shape[:2]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize tracking points in ROI only
    roi_mask = get_roi_mask(h, w)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=300, qualityLevel=0.2, 
                                  minDistance=7, blockSize=7, mask=roi_mask)

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    overspeed_cooldown = 0
    lane_cooldown = 0
    
    # WARM-UP: Skip violations for first N frames to let detection stabilize
    frame_count = 0
    WARMUP_FRAMES = 60  # ~2 seconds at 30fps - prevents false violations at video start

    # Conservative lane type stability (prevents transient mis-detections)
    right_solid_streak = 0
    left_yellow_solid_streak = 0
    REQUIRED_SOLID_STREAK = 15  # INCREASED from 8 - must be stable for 15 frames (0.5 sec at 30fps)

    # Periodically refresh DB config so admin changes apply live
    last_cfg_pull = 0.0
    CFG_PULL_SEC = 5.0

    # Speed smoothing (EMA)
    smoothed_speed = 0.0
    
    # === ADVANCED SPEED SMOOTHING ===
    # Use a buffer of recent speed readings for more stable output
    from collections import deque
    SPEED_BUFFER_SIZE = 30  # ~1 second of readings at 30fps
    speed_buffer = deque(maxlen=SPEED_BUFFER_SIZE)
    # ================================

    print("[INFO] Road monitor started (improved optical flow). Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        frame_count += 1  # Track frame number for warm-up period
        if now - last_cfg_pull >= CFG_PULL_SEC:
            cfg = load_vehicle_config(conn, vehicle_id)
            admin_speed_limit = cfg.get("speed_limit_kmh") or admin_speed_limit
            lane_band_px = cfg.get("lane_sensitivity_px") or lane_band_px
            last_cfg_pull = now

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # ==========================================
        # 1) YOLO SIGN DETECTION (speed + junction)
        # ==========================================
        dets = sign_model.detect(frame)

        # Speed limit voting (with geometry filter)
        # --- FILTER DETECTIONS ---
        # Focused model only outputs speed + junction classes, so filtering is simpler.
        final_dets = []
        speed_changed = False

        for d in dets:
            name = d["cls_name"]
            keep = True
            
            print(f"[DEBUG-YOLO] Saw {name} Conf:{d['conf']:.2f}")

            if name in SPEED_VALUE:
                # Geometry check
                if not is_valid_speed_sign_box(d, w, h):
                    print(f"[DEBUG-REJECT] {name} failed GEOMETRY check.")
                    keep = False
                # Circularity check — reject diamond/square shapes
                elif not verify_speed_sign_circle(frame, d):
                    print(f"[DEBUG-REJECT] {name} failed CIRCLE check (not round — likely warning/priority sign).")
                    keep = False
                else:
                    # Directly use YOLO class
                    if sign_state.update_speed_limit(SPEED_VALUE[name]):
                        speed_changed = True
                    print(f"[DEBUG-KEEP] {name} conf={d['conf']:.2f} VALIDATED.")
                    
            elif name in JUNCTION_CLASSES:
                if not is_valid_junction_box(d, w, h):
                    keep = False
                else:
                    sign_state.update_junction_vote(name)
                     
            if keep:
                final_dets.append(d)
                
        dets = final_dets
        # --------------------------

        if speed_changed:
            active = sign_state.get_current_speed_limit()
            detail = f"Speed limit updated to {active} km/h (YOLO)."
            log_event(cursor, conn, "speed_limit_update", vehicle_id, float(active), detail)
            
            # --- HUD BANNER FOR VISUAL CONFIRMATION ---
            # Draw a green banner at the top center
            cv2.rectangle(frame, (w//2 - 250, 50), (w//2 + 250, 100), (0, 255, 0), -1)
            cv2.putText(frame, f"SPEED LIMIT UPDATED: {active} KM/H", (w//2 - 230, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            # ------------------------------------------

        # ---- Junction voting with strong filters ----
        # This loop is now redundant as junction voting happens in the filtering block above.
        # However, the original code had a separate loop for junctions.
        # If the intent is to only update sign_state once per frame for each type,
        # the filtering loop already handles it.
        # Keeping this commented out for now, assuming the filtering loop is the source of truth.
        # for d in dets: # 'dets' is now 'final_dets'
        #     if d["cls_name"] in JUNCTION_CLASSES and is_valid_junction_box(d, w, h):
        #         sign_state.update_junction_vote(d["cls_name"])

        stable_junction = sign_state.pop_stable_junction()
        if stable_junction and sign_state.can_alert_junction():
            detail = f"Junction sign detected: {stable_junction}"
            # NOTE: Junction logging disabled - only TTS alert, no DB entry
            # log_event(cursor, conn, "junction_ahead", vehicle_id, 1.0, detail)
            
            # --- HUD BANNER FOR VISUAL CONFIRMATION ---
            # Draw a yellow banner for warnings
            cv2.rectangle(frame, (w//2 - 300, 110), (w//2 + 300, 160), (0, 255, 255), -1)
            # GENERIC LABEL: "JUNCTION AHEAD" covers both T-Junction and Cross-Roads
            cv2.putText(frame, "CAUTION: JUNCTION AHEAD", (w//2 - 280, 145), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            # ------------------------------------------
            
            if engine is not None:
                try:
                    # Generic speech for all junction types
                    engine.say("Junction ahead")
                    engine.runAndWait()
                except Exception:
                    pass

        # ==========================================
        # 2) SPEED ESTIMATION (improved optical flow)
        # ==========================================
        
        # === ROAD VISIBILITY CHECK ===
        # If the camera is pointing at trees/sky, skip optical flow
        # and retain the last good speed reading
        road_visible = check_road_visible(frame, h, w)
        
        if road_visible:
            # Road is visible - calculate optical flow
            flow_value, p0 = compute_optical_flow_speed(old_gray, frame_gray, p0, lk_params, h, w)
            old_gray = frame_gray.copy()
            
            # Calculate raw speed from flow
            raw_speed = (flow_value * kmh_per_pixel) + speed_offset
            if raw_speed < 0:
                raw_speed = 0.0
        else:
            # Road NOT visible - skip optical flow, use last good speed
            old_gray = frame_gray.copy()
            # raw_speed stays unchanged from previous frame
            if 'raw_speed' not in dir() or smoothed_speed == 0:
                raw_speed = 0.0
            else:
                raw_speed = smoothed_speed  # Hold last good speed
        # ==============================
        
        # === ADVANCED SMOOTHING WITH OUTLIER REJECTION ===
        # Only accept readings that are within 30% of current median (STRICTER)
        if len(speed_buffer) >= 5:
            current_median = np.median(list(speed_buffer))
            # Reject outliers: readings too far from current median
            if current_median > 5:  # Only apply when moving
                if abs(raw_speed - current_median) > current_median * 0.30:
                    # Outlier detected - use current median instead
                    raw_speed = current_median
        
        # Add to buffer
        speed_buffer.append(raw_speed)
        
        # Use median of buffer for display (very stable)
        if len(speed_buffer) >= 3:
            buffer_median = np.median(list(speed_buffer))
        else:
            buffer_median = raw_speed
        
        # Also apply EMA on top of buffer median for extra smoothness
        smoothed_speed = SPEED_SMOOTHING_ALPHA * buffer_median + (1 - SPEED_SMOOTHING_ALPHA) * smoothed_speed
        speed_kmh = smoothed_speed
        # ==================================================

        # DEBUG SPEED
        # if frame_count % 30 == 0:
        #    print(f"[DEBUG] Speed: {speed_kmh:.1f} km/h (Flow: {flow_value:.2f} Raw: {raw_speed:.1f})")
        # ==========================================
        # 3) SPEED LIMIT PRIORITY LOGIC (FIXED)
        # ==========================================

        # Priority (FIXED):
        #   (A) Detected Sign (YOLO) - takes priority when sign is seen
        #   (B) Admin/parent override (DB config) - only if no sign detected
        #   (C) Default Fallback (Only if nothing else)
        
        yolo_limit = sign_state.get_current_speed_limit()
        
        # YOLO detected sign takes priority
        if yolo_limit is not None:
             active_speed_limit = int(yolo_limit)
        elif cfg.get("speed_limit_kmh") is not None:
             active_speed_limit = int(cfg.get("speed_limit_kmh"))
        else:
             active_speed_limit = DEFAULT_SPEED_LIMIT_KMH

        # ==========================================
        # 4) OVERSPEED DETECTION WITH EVIDENCE
        # ==========================================
        if overspeed_cooldown > 0:
            overspeed_cooldown -= 1
        
        # Check for overspeed violation
        if speed_kmh > active_speed_limit + OVERSPEED_TOLERANCE_KMH and overspeed_cooldown == 0:
            # Capture evidence frame with speed overlay
            ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"overspeed_{vehicle_id}_{ts_file}.jpg"
            filepath = os.path.join(EVIDENCE_DIR, filename)
            
            # Draw speed info on evidence
            evidence_frame = frame.copy()
            evidence_frame = YOLOSignDetector.draw(evidence_frame, dets)
            
            # Add speed overlay box
            cv2.rectangle(evidence_frame, (20, 20), (350, 100), (0, 0, 200), -1)
            cv2.putText(evidence_frame, f"OVERSPEED: {speed_kmh:.1f} km/h", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(evidence_frame, f"Limit: {active_speed_limit} km/h", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imwrite(filepath, evidence_frame)
            
            detail = f"Overspeed: {speed_kmh:.1f} km/h (limit: {active_speed_limit} km/h). evidence={filename}"
            log_event(cursor, conn, "overspeed", vehicle_id, float(speed_kmh), detail, image_path=filename)
            
            if engine is not None:
                try:
                    engine.say("Overspeed warning")
                    engine.runAndWait()
                except Exception:
                    pass
            
            overspeed_cooldown = OVERSPEED_COOLDOWN_FRAMES

        # ==========================================
        # 5) LANE DEVIATION (ONLY THIS SAVES IMAGE)
        # ==========================================
        # Use DL lane detector if available, else fallback to traditional CV
        
        if USE_DL_LANE_DETECTOR:
            lane_info = dl_lane_detector.detect(frame)
        else:
            lane_info = detect_lane_lines(frame)

        cam_center_x = w // 2  # Vehicle position approximated by camera center
        lane_center_x = None

        left = lane_info.get("left", {})           # White center line
        right = lane_info.get("right", {})         # Right edge (optional)
        yellow_edge = lane_info.get("yellow_edge", {})  # Yellow left edge
        
        left_x = left.get("x_bottom")              # White center line x position
        right_x = right.get("x_bottom")            # Right edge x position
        yellow_x = yellow_edge.get("x_bottom")     # Yellow left edge x position
        
        left_type = left.get("type")
        right_type = right.get("type")
        yellow_type = yellow_edge.get("type")
        left_color = left.get("color", "unknown")

        # Calculate lane center (ORIGINAL APPROACH)
        if left_x is not None and right_x is not None:
            lane_center_x = int((left_x + right_x) / 2)
        elif left_x is not None:
            lane_center_x = left_x
        elif right_x is not None:
            lane_center_x = right_x

        violation_side = None
        violation_reason = None
        deviation = 0

        if lane_center_x is not None:
            # ORIGINAL FORMULA: deviation = cam_center - lane_center
            # deviation > 0 means vehicle is to the RIGHT of lane center
            # deviation < 0 means vehicle is to the LEFT of lane center
            deviation = cam_center_x - lane_center_x

            # === LANE VIOLATION DETECTION (SRI LANKAN LEFT-HAND TRAFFIC) ===
            # In Sri Lanka, vehicles drive on the LEFT side of the road
            # The WHITE CENTER LINE is on the RIGHT side of the vehicle
            # The YELLOW EDGE is on the LEFT side of the vehicle
            
            if lane_cooldown == 0 and frame_count > WARMUP_FRAMES:
                # Sensitivity threshold
                LANE_SENSITIVITY_PX = 50  # Pixels past the line to trigger
                
                # VIOLATION 1: Crossed WHITE CENTER LINE (overtaking on wrong side)
                # Vehicle center moves too far RIGHT past the white center line
                center_line_detected = (left_x is not None)
                
                if center_line_detected:
                    if cam_center_x > left_x + LANE_SENSITIVITY_PX:
                        violation_side = "right"
                        violation_reason = "solid_centerline_overtake"
                        print(f"[VIOLATION] Crossed center line! cam={cam_center_x} > line={left_x}")
                
                # VIOLATION 2: Crossed YELLOW LEFT EDGE (going off-road to the left)
                # Vehicle center moves too far LEFT past the yellow edge
                yellow_edge_detected = (yellow_x is not None)
                
                if yellow_edge_detected:
                    # Yellow edge is on LEFT of vehicle
                    # Vehicle normal position: cam_center_x > yellow_x
                    # VIOLATION: cam_center_x < yellow_x (vehicle crossed over yellow to the left)
                    if cam_center_x < yellow_x + LANE_SENSITIVITY_PX:
                        violation_side = "left"
                        violation_reason = "yellow_edge_cross"
                        print(f"[VIOLATION] Crossed yellow edge! cam={cam_center_x} < edge={yellow_x}")

        if lane_cooldown > 0:
            lane_cooldown -= 1

        if violation_side is not None and lane_cooldown == 0:
            ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lane_violation_{vehicle_id}_{ts_file}.jpg"
            filepath = os.path.join(EVIDENCE_DIR, filename)

            # Draw lanes on evidence image
            if USE_DL_LANE_DETECTOR:
                annotated = dl_lane_detector.draw_lanes(frame.copy(), lane_info)
            else:
                annotated = draw_lane_lines(frame.copy(), lane_info, highlight=violation_side)
            annotated = YOLOSignDetector.draw(annotated, dets)  # include sign boxes for context
            
            # Draw center line for visualization
            cv2.line(annotated, (cam_center_x, h), (cam_center_x, h // 2), (255, 0, 255), 2)
            cv2.putText(annotated, "CAM CENTER", (cam_center_x - 40, h // 2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            cv2.imwrite(filepath, annotated)

            detail = (
                f"Lane violation {violation_side} ({violation_reason}). "
                f"cam_x={cam_center_x}, right_x={right_x}, left_x={left_x}. "
                f"Left={left_type}/{left_color} Right={right_type}. evidence={filename}"
            )
            log_event(cursor, conn, "lane_deviation", vehicle_id, float(cam_center_x), detail, image_path=filename)

            if engine is not None:
                try:
                    engine.say("Lane violation detected")
                    engine.runAndWait()
                except Exception:
                    pass

            lane_cooldown = LANE_COOLDOWN_FRAMES

        # ==========================================
        # 6) DISPLAY (debug)
        # ==========================================
        disp = frame.copy()
        disp = YOLOSignDetector.draw(disp, dets)
        
        # Draw lane lines with appropriate method
        try:
            if USE_DL_LANE_DETECTOR:
                disp = dl_lane_detector.draw_lanes(disp, lane_info)
            else:
                disp = draw_lane_lines(disp, lane_info, highlight=violation_side)
        except Exception as e:
            # Handle any drawing errors gracefully
            cv2.putText(disp, f"Lane draw err: {type(e).__name__}", (20, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(
            disp,
            f"Speed: {speed_kmh:.1f} km/h | Limit: {active_speed_limit} km/h",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if speed_kmh <= active_speed_limit else (0, 0, 255),
            2,
        )

        if stable_junction:
            cv2.putText(
                disp,
                "JUNCTION: JUNCTION AHEAD",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        cv2.putText(
            disp,
            f"LaneSens(px): {lane_band_px} | AdminLimit: {admin_speed_limit}",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        if stats_callback is not None:
            try:
                stats_callback({
                    'speed_kmh': round(speed_kmh, 1),
                    'active_limit': active_speed_limit,
                    'lane_sens': lane_band_px,
                    'admin_limit': admin_speed_limit,
                    'is_junction': bool(stable_junction)
                })
            except Exception:
                pass

        # Show - FULLSCREEN for panel presentation (skip in headless mode)
        if not headless:
            # Create fullscreen window on first frame
            if frame_count == 1:
                cv2.namedWindow("Road Monitor", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Road Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Road Monitor", disp)
            # out_writer.write(disp)

            # Wait key with slow-motion support
            # Press 's' to toggle slow mode, 'q' to quit
            wait_time = 200 if getattr(run_monitor, 'slow_mode', False) else 1
            key = cv2.waitKey(wait_time)
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Toggle slow mode
                run_monitor.slow_mode = not getattr(run_monitor, 'slow_mode', False)
                print(f"[INFO] Slow mode: {'ON (200ms/frame)' if run_monitor.slow_mode else 'OFF'}")

    cap.release()
    # out_writer.release()
    if not headless:
        cv2.destroyAllWindows()
    conn.close()
    print(f"[INFO] Monitoring stopped.")


# =========================
# ---- Entry point --------
# =========================

if __name__ == "__main__":
    # Usage:
    #   python road_monitor.py 0 BUS_01
    #   python road_monitor.py videos/test.mp4 BUS_01
    #   python road_monitor.py --calibrate videos/calib_30_60.mp4

    mode = "run"
    video_source = 0
    vehicle_id = "unknown"

    if len(sys.argv) > 1:
        if sys.argv[1] == "--calibrate":
            mode = "calibrate"
            if len(sys.argv) > 2:
                video_source = sys.argv[2]
        else:
            video_source = sys.argv[1]
            if len(sys.argv) > 2:
                vehicle_id = sys.argv[2]

    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    if mode == "calibrate":
        run_calibration(video_source)
    else:
        run_monitor(video_source, vehicle_id=vehicle_id)
