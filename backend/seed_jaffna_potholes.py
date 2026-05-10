"""Seed Jaffna road potholes into the map database."""
import sqlite3, time, os

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "safety_events.db")

# Jaffna–Sangupiddy Road & Jaffna–Kurikkaduwan Road potholes
# severity: 1=low/small, 2=medium
potholes = [
    # Jaffna → Sangupiddy Road (heading southwest along A32)
    (9.6578, 80.0130, 2, "Jaffna–Sangupiddy Rd km 1"),
    (9.6482, 80.0048, 1, "Jaffna–Sangupiddy Rd km 3"),
    (9.6371, 79.9962, 2, "Jaffna–Sangupiddy Rd km 5"),
    (9.6250, 79.9905, 1, "Jaffna–Sangupiddy Rd km 7"),
    (9.6112, 79.9878, 2, "Jaffna–Sangupiddy Rd km 9"),
    (9.5991, 79.9851, 1, "Sangupiddy approach"),
    # Jaffna → Kurikkaduwan Road (heading west-northwest)
    (9.6672, 79.9975, 2, "Jaffna–Kurikkaduwan Rd km 2"),
    (9.6648, 79.9830, 1, "Jaffna–Kurikkaduwan Rd km 4"),
    (9.6601, 79.9685, 2, "Jaffna–Kurikkaduwan Rd km 6"),
    (9.6523, 79.9542, 1, "Jaffna–Kurikkaduwan Rd km 8"),
    (9.6435, 79.9408, 2, "Kurikkaduwan approach"),
]

conn = sqlite3.connect(DB)
conn.execute("""
    CREATE TABLE IF NOT EXISTS potholes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude REAL, longitude REAL,
        severity REAL, timestamp REAL,
        vehicle_id TEXT, source TEXT
    )
""")
for col in ("vehicle_id TEXT", "source TEXT"):
    try:
        conn.execute(f"ALTER TABLE potholes ADD COLUMN {col}")
    except Exception:
        pass
conn.commit()

# Remove any previously seeded rows (idempotent)
conn.execute("DELETE FROM potholes WHERE source LIKE 'seed_%'")
conn.commit()

ts = time.time()
for lat, lon, sev, label in potholes:
    conn.execute(
        "INSERT INTO potholes (latitude, longitude, severity, timestamp, vehicle_id, source) "
        "VALUES (?,?,?,?,?,?)",
        (lat, lon, sev, ts, "SURVEY-JAFFNA", f"seed_{label}")
    )
conn.commit()
count = conn.execute("SELECT COUNT(*) FROM potholes WHERE source LIKE 'seed_%'").fetchone()[0]
print(f"[OK] Inserted {count} seeded potholes for Jaffna roads")
conn.close()
