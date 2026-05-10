# api.py - Road Rule Violation Detection System API
# Updated: 2026-01-07 - Added user management, JWT auth, full CRUD

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import sqlite3
from datetime import datetime, timedelta
import hashlib
import secrets
import os

DB_PATH = "new.db"
EVIDENCE_DIR = "evidence"

# Ensure evidence directory exists
os.makedirs(EVIDENCE_DIR, exist_ok=True)

app = Flask(__name__, static_folder='frontend')
CORS(app)

# ============================================================
# TOKEN MANAGEMENT (Simple token-based auth)
# ============================================================
tokens = {}  # token -> {user_id, role, expires}

def generate_token(user_id, role):
    """Generate auth token valid for 24 hours"""
    token = secrets.token_hex(32)
    tokens[token] = {
        "user_id": user_id,
        "role": role,
        "expires": datetime.now() + timedelta(hours=24)
    }
    return token

def verify_token(token):
    """Returns user info if valid token, else None"""
    if not token or token not in tokens:
        return None
    info = tokens[token]
    if datetime.now() > info["expires"]:
        del tokens[token]
        return None
    return info

def get_current_user():
    """Get current user from Authorization header"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        return verify_token(token)
    return None

def require_auth(f):
    """Decorator to require authentication"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return decorated

def require_admin(f):
    """Decorator to require admin role"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        if user["role"] != "admin":
            return jsonify({"error": "Admin access required"}), 403
        request.current_user = user
        return f(*args, **kwargs)
    return decorated

def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

# ============================================================
# DATABASE
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            role TEXT DEFAULT 'user',
            created_at TEXT,
            updated_at TEXT
        )
    """)

    # Events table (road_monitor writes here)
    cur.execute("""
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

    # Add image_path column if it doesn't exist
    try:
        cur.execute("ALTER TABLE events_v2 ADD COLUMN image_path TEXT")
    except:
        pass  # Column already exists

    # Vehicles table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
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
        )
    """)

    # Add owner_id column if it doesn't exist
    try:
        cur.execute("ALTER TABLE vehicles ADD COLUMN owner_id INTEGER")
    except:
        pass

    # Create default admin if not exists
    cur.execute("SELECT id FROM users WHERE email = 'admin@system.local'")
    if not cur.fetchone():
        cur.execute("""
            INSERT INTO users (email, password_hash, name, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ('admin@system.local', hash_password('Admin'), 'System Admin', 'admin', datetime.now().isoformat()))

    conn.commit()
    conn.close()

init_db()

# ============================================================
# STATIC FILES (Serve frontend)
# ============================================================
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

# ============================================================
# AUTH ENDPOINTS
# ============================================================
@app.route("/api/auth/register", methods=["POST"])
def register():
    """Register new user"""
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()
    name = (data.get("name") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    conn = get_db()
    cur = conn.cursor()

    # Check if email exists
    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cur.fetchone():
        conn.close()
        return jsonify({"error": "Email already registered"}), 409

    now = datetime.now().isoformat()
    cur.execute("""
        INSERT INTO users (email, password_hash, name, role, created_at, updated_at)
        VALUES (?, ?, ?, 'user', ?, ?)
    """, (email, hash_password(password), name, now, now))
    user_id = cur.lastrowid
    conn.commit()
    conn.close()

    token = generate_token(user_id, "user")
    return jsonify({"ok": True, "token": token, "role": "user", "user_id": user_id}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    """Login user or admin"""
    data = request.get_json(force=True)
    
    # Check for Admin/Admin special case
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    email = (data.get("email") or "").strip().lower()

    # Admin special login
    if username == "Admin" and password == "Admin":
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1")
        row = cur.fetchone()
        conn.close()
        admin_id = row["id"] if row else 1
        token = generate_token(admin_id, "admin")
        return jsonify({"ok": True, "token": token, "role": "admin"})

    # Regular user login
    if not email:
        email = username  # Allow username to be email
    
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash, role, name FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()

    if not row or row["password_hash"] != hash_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = generate_token(row["id"], row["role"])
    return jsonify({
        "ok": True, 
        "token": token, 
        "role": row["role"],
        "user_id": row["id"],
        "name": row["name"]
    })

@app.route("/api/auth/me", methods=["GET"])
@require_auth
def get_me():
    """Get current user info"""
    user = request.current_user
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, email, name, role, created_at FROM users WHERE id = ?", (user["user_id"],))
    row = cur.fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "User not found"}), 404
    return jsonify(dict(row))

# ============================================================
# USER - VEHICLE MANAGEMENT (Own vehicles only)
# ============================================================
@app.route("/api/user/vehicles", methods=["GET", "POST"])
@require_auth
def user_vehicles():
    user = request.current_user
    conn = get_db()
    cur = conn.cursor()

    if request.method == "POST":
        data = request.get_json(force=True)
        vehicle_id = (data.get("vehicle_id") or "").strip()
        label = (data.get("label") or "").strip()
        notes = (data.get("notes") or "").strip()
        speed_limit = data.get("speed_limit_kmh")

        if not vehicle_id:
            conn.close()
            return jsonify({"error": "Vehicle ID required"}), 400

        now = datetime.now().isoformat()
        try:
            cur.execute("""
                INSERT INTO vehicles (vehicle_id, owner_id, label, notes, speed_limit_kmh, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (vehicle_id, user["user_id"], label, notes, speed_limit, now, now))
            conn.commit()
            new_id = cur.lastrowid
            conn.close()
            return jsonify({"ok": True, "id": new_id}), 201
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({"error": "Vehicle ID already exists"}), 409

    # GET - list user's vehicles
    cur.execute("""
        SELECT v.*, 
               (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id) as total_events,
               (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id AND e.type = 'lane_deviation') as lane_count,
               (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id AND e.type = 'overspeed') as overspeed_count
        FROM vehicles v
        WHERE v.owner_id = ?
        ORDER BY v.created_at DESC
    """, (user["user_id"],))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route("/api/user/vehicles/<int:id>", methods=["GET", "PUT", "DELETE"])
@require_auth
def user_vehicle_detail(id):
    user = request.current_user
    conn = get_db()
    cur = conn.cursor()

    # Check ownership
    cur.execute("SELECT * FROM vehicles WHERE id = ? AND owner_id = ?", (id, user["user_id"]))
    vehicle = cur.fetchone()
    if not vehicle:
        conn.close()
        return jsonify({"error": "Vehicle not found"}), 404

    if request.method == "DELETE":
        cur.execute("DELETE FROM vehicles WHERE id = ?", (id,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})

    if request.method == "PUT":
        data = request.get_json(force=True)
        fields = []
        params = []
        
        for field in ["label", "notes", "speed_limit_kmh", "lane_sensitivity_px"]:
            if field in data:
                fields.append(f"{field} = ?")
                params.append(data[field])
        
        if fields:
            fields.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(id)
            cur.execute(f"UPDATE vehicles SET {', '.join(fields)} WHERE id = ?", params)
            conn.commit()
        
        conn.close()
        return jsonify({"ok": True})

    # GET
    conn.close()
    return jsonify(dict(vehicle))

@app.route("/api/user/vehicles/<int:id>/events", methods=["GET"])
@require_auth
def user_vehicle_events(id):
    user = request.current_user
    conn = get_db()
    cur = conn.cursor()

    # Check ownership
    cur.execute("SELECT vehicle_id FROM vehicles WHERE id = ? AND owner_id = ?", (id, user["user_id"]))
    vehicle = cur.fetchone()
    if not vehicle:
        conn.close()
        return jsonify({"error": "Vehicle not found"}), 404

    limit = int(request.args.get("limit", 100))
    cur.execute("""
        SELECT * FROM events_v2 
        WHERE vehicle_id = ?
        ORDER BY ts DESC
        LIMIT ?
    """, (vehicle["vehicle_id"], limit))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)

# ============================================================
# ADMIN - USER MANAGEMENT
# ============================================================
@app.route("/api/admin/users", methods=["GET", "POST"])
@require_admin
def admin_users():
    conn = get_db()
    cur = conn.cursor()

    if request.method == "POST":
        data = request.get_json(force=True)
        email = (data.get("email") or "").strip().lower()
        password = (data.get("password") or "").strip()
        name = (data.get("name") or "").strip()
        role = data.get("role", "user")

        if not email or not password:
            conn.close()
            return jsonify({"error": "Email and password required"}), 400

        now = datetime.now().isoformat()
        try:
            cur.execute("""
                INSERT INTO users (email, password_hash, name, role, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (email, hash_password(password), name, role, now, now))
            conn.commit()
            new_id = cur.lastrowid
            conn.close()
            return jsonify({"ok": True, "id": new_id}), 201
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({"error": "Email already exists"}), 409

    # GET - list all users with vehicle counts
    cur.execute("""
        SELECT u.*, 
               (SELECT COUNT(*) FROM vehicles v WHERE v.owner_id = u.id) as vehicle_count
        FROM users u
        ORDER BY u.created_at DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    # Remove password hash from response
    for row in rows:
        row.pop("password_hash", None)
    conn.close()
    return jsonify(rows)

@app.route("/api/admin/users/<int:id>", methods=["GET", "PUT", "DELETE"])
@require_admin
def admin_user_detail(id):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE id = ?", (id,))
    user = cur.fetchone()
    if not user:
        conn.close()
        return jsonify({"error": "User not found"}), 404

    if request.method == "DELETE":
        # Delete user's vehicles first
        cur.execute("DELETE FROM vehicles WHERE owner_id = ?", (id,))
        cur.execute("DELETE FROM users WHERE id = ?", (id,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})

    if request.method == "PUT":
        data = request.get_json(force=True)
        fields = []
        params = []
        
        for field in ["name", "email", "role"]:
            if field in data:
                fields.append(f"{field} = ?")
                params.append(data[field])
        
        if "password" in data and data["password"]:
            fields.append("password_hash = ?")
            params.append(hash_password(data["password"]))
        
        if fields:
            fields.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(id)
            cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", params)
            conn.commit()
        
        conn.close()
        return jsonify({"ok": True})

    # GET
    result = dict(user)
    result.pop("password_hash", None)
    
    # Include user's vehicles
    cur.execute("SELECT * FROM vehicles WHERE owner_id = ?", (id,))
    result["vehicles"] = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(result)

# ============================================================
# ADMIN - VEHICLE MANAGEMENT (All vehicles)
# ============================================================
@app.route("/api/admin/vehicles", methods=["GET", "POST"])
@require_admin
def admin_vehicles():
    conn = get_db()
    cur = conn.cursor()

    if request.method == "POST":
        data = request.get_json(force=True)
        vehicle_id = (data.get("vehicle_id") or "").strip()
        owner_id = data.get("owner_id")
        label = (data.get("label") or "").strip()
        notes = (data.get("notes") or "").strip()

        if not vehicle_id:
            conn.close()
            return jsonify({"error": "Vehicle ID required"}), 400

        now = datetime.now().isoformat()
        try:
            cur.execute("""
                INSERT INTO vehicles (vehicle_id, owner_id, label, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (vehicle_id, owner_id, label, notes, now, now))
            conn.commit()
            new_id = cur.lastrowid
            conn.close()
            return jsonify({"ok": True, "id": new_id}), 201
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({"error": "Vehicle ID already exists"}), 409

    # GET - list all vehicles with owner info
    cur.execute("""
        SELECT v.*, u.name as owner_name, u.email as owner_email,
               (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id) as total_events
        FROM vehicles v
        LEFT JOIN users u ON v.owner_id = u.id
        ORDER BY v.created_at DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route("/api/admin/vehicles/<int:id>", methods=["GET", "PUT", "DELETE"])
@require_admin
def admin_vehicle_detail(id):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM vehicles WHERE id = ?", (id,))
    vehicle = cur.fetchone()
    if not vehicle:
        conn.close()
        return jsonify({"error": "Vehicle not found"}), 404

    if request.method == "DELETE":
        cur.execute("DELETE FROM vehicles WHERE id = ?", (id,))
        conn.commit()
        conn.close()
        return jsonify({"ok": True})

    if request.method == "PUT":
        data = request.get_json(force=True)
        fields = []
        params = []
        
        for field in ["vehicle_id", "owner_id", "label", "notes", "speed_limit_kmh", 
                      "lane_sensitivity_px", "kmh_per_pixel"]:
            if field in data:
                fields.append(f"{field} = ?")
                params.append(data[field])
        
        if fields:
            fields.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(id)
            cur.execute(f"UPDATE vehicles SET {', '.join(fields)} WHERE id = ?", params)
            conn.commit()
        
        conn.close()
        return jsonify({"ok": True})

    # GET
    result = dict(vehicle)
    # Get owner info
    if vehicle["owner_id"]:
        cur.execute("SELECT name, email FROM users WHERE id = ?", (vehicle["owner_id"],))
        owner = cur.fetchone()
        if owner:
            result["owner_name"] = owner["name"]
            result["owner_email"] = owner["email"]
    conn.close()
    return jsonify(result)

@app.route("/api/admin/vehicles/<int:id>/calibration", methods=["PUT"])
@require_admin
def admin_vehicle_calibration(id):
    """Save calibration value to vehicle"""
    conn = get_db()
    cur = conn.cursor()
    
    data = request.get_json(force=True)
    kmh_per_pixel = data.get("kmh_per_pixel")
    
    if kmh_per_pixel is None:
        conn.close()
        return jsonify({"error": "kmh_per_pixel required"}), 400
    
    cur.execute("""
        UPDATE vehicles SET kmh_per_pixel = ?, updated_at = ?
        WHERE id = ?
    """, (kmh_per_pixel, datetime.now().isoformat(), id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# ============================================================
# VIDEO UPLOAD CALIBRATION
# ============================================================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/api/admin/calibration/upload", methods=["POST"])
@require_admin
def calibration_upload():
    """
    Upload 30 and 60 kmph videos for auto-calibration.
    Expects multipart form with 'video_30' and 'video_60' files.
    """
    if 'video_30' not in request.files or 'video_60' not in request.files:
        return jsonify({"error": "Both video_30 and video_60 files required"}), 400
    
    video_30 = request.files['video_30']
    video_60 = request.files['video_60']
    
    if video_30.filename == '' or video_60.filename == '':
        return jsonify({"error": "Both files must be selected"}), 400
    
    # Save uploaded files
    import time
    timestamp = int(time.time())
    path_30 = os.path.join(UPLOAD_DIR, f"calib_30_{timestamp}.mp4")
    path_60 = os.path.join(UPLOAD_DIR, f"calib_60_{timestamp}.mp4")
    
    video_30.save(path_30)
    video_60.save(path_60)
    
    try:
        # Run calibration using auto_calibrate module
        from auto_calibrate import process_video
        
        print(f"[CALIBRATION API] Processing 30 km/h video: {path_30}")
        flow_30, std_30, n_30 = process_video(path_30, 30, skip_start_frames=30, skip_end_frames=30)
        
        print(f"[CALIBRATION API] Processing 60 km/h video: {path_60}")
        flow_60, std_60, n_60 = process_video(path_60, 60, skip_start_frames=30, skip_end_frames=30)
        
        if flow_30 is None or flow_60 is None:
            return jsonify({
                "error": "Calibration failed - could not process videos. Check video quality."
            }), 400
        
        # Calculate kmh_per_pixel using two-point calibration
        flow_diff = flow_60 - flow_30
        if flow_diff < 0.1:
            return jsonify({
                "error": "Flow difference too small. Ensure proper speed difference."
            }), 400
        
        kmh_per_pixel = 30.0 / flow_diff
        offset = 30.0 - flow_30 * kmh_per_pixel
        
        # Cleanup uploaded files
        try:
            os.remove(path_30)
            os.remove(path_60)
        except:
            pass
        
        return jsonify({
            "ok": True,
            "kmh_per_pixel": round(kmh_per_pixel, 4),
            "offset": round(offset, 4),
            "flow_30": round(flow_30, 4),
            "flow_60": round(flow_60, 4),
            "samples_30": n_30,
            "samples_60": n_60
        })
        
    except Exception as e:
        try:
            os.remove(path_30)
            os.remove(path_60)
        except:
            pass
        print(f"[CALIBRATION API ERROR] {str(e)}")
        return jsonify({"error": f"Calibration error: {str(e)}"}), 500


# ============================================================
# GLOBAL STATS
# ============================================================
@app.route("/api/summary")
def global_summary():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) AS total_events,
            SUM(CASE WHEN type = 'lane_deviation' THEN 1 ELSE 0 END) AS lane_count,
            SUM(CASE WHEN type = 'overspeed' THEN 1 ELSE 0 END) AS overspeed_count,
            COUNT(DISTINCT vehicle_id) AS active_vehicles
        FROM events_v2
    """)
    row = cur.fetchone()
    data = dict(row) if row else {}

    cur.execute("SELECT COUNT(*) AS user_count FROM users WHERE role = 'user'")
    rv = cur.fetchone()
    data["user_count"] = rv["user_count"] if rv else 0

    cur.execute("SELECT COUNT(*) AS vehicle_count FROM vehicles")
    rv = cur.fetchone()
    data["vehicle_count"] = rv["vehicle_count"] if rv else 0

    conn.close()
    return jsonify(data)

# ============================================================
# EVIDENCE IMAGES
# ============================================================
@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    return send_from_directory(EVIDENCE_DIR, filename)

# ============================================================
# LEGACY ENDPOINTS (Keep for backward compatibility)
# ============================================================
@app.route("/api/login", methods=["POST"])
def legacy_login():
    """Legacy admin login endpoint"""
    data = request.get_json(force=True)
    u = (data.get("username") or "").strip()
    p = (data.get("password") or "").strip()
    if u == "admin" and p == "admin":
        return jsonify({"ok": True, "role": "admin"})
    return jsonify({"ok": False, "error": "Invalid credentials"}), 401

@app.route("/api/vehicles", methods=["GET", "POST"])
def legacy_vehicles():
    """Legacy vehicles endpoint (admin access)"""
    conn = get_db()
    cur = conn.cursor()

    if request.method == "POST":
        data = request.get_json(force=True)
        vehicle_id = (data.get("vehicle_id") or "").strip()
        label = (data.get("label") or "").strip()
        
        if not vehicle_id:
            conn.close()
            return jsonify({"error": "vehicle_id is required"}), 400

        now = datetime.now().isoformat()
        try:
            cur.execute("""
                INSERT INTO vehicles (vehicle_id, label, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (vehicle_id, label, now, now))
            conn.commit()
            conn.close()
            return jsonify({"status": "ok"}), 201
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({"error": "vehicle_id already exists"}), 409

    # GET
    cur.execute("""
        SELECT v.*, 
               (SELECT COUNT(*) FROM events_v2 e WHERE e.vehicle_id = v.vehicle_id) as total_events
        FROM vehicles v
        ORDER BY v.vehicle_id
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route("/api/vehicles/<vehicle_id>/config", methods=["GET", "POST"])
def legacy_vehicle_config(vehicle_id):
    """Legacy vehicle config endpoint"""
    conn = get_db()
    cur = conn.cursor()

    if request.method == "GET":
        cur.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "vehicle not found"}), 404
        return jsonify(dict(row))

    data = request.get_json(force=True)
    fields = []
    params = []

    for field in ["speed_limit_kmh", "lane_sensitivity_px", "label", "notes"]:
        if field in data:
            fields.append(f"{field} = ?")
            params.append(data[field])

    if fields:
        fields.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(vehicle_id)
        cur.execute(f"UPDATE vehicles SET {', '.join(fields)} WHERE vehicle_id = ?", params)
        conn.commit()

    conn.close()
    return jsonify({"status": "ok"})

@app.route("/api/vehicles/<vehicle_id>/events")
def legacy_vehicle_events(vehicle_id):
    """Legacy vehicle events endpoint"""
    limit = int(request.args.get("limit", 50))
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM events_v2
        WHERE vehicle_id = ?
        ORDER BY ts DESC
        LIMIT ?
    """, (vehicle_id, limit))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("Road Monitor API running on http://localhost:5000")
    print("Serving frontend from ./frontend/")
    app.run(host="0.0.0.0", port=5000, debug=True)
