"""
reset_detections.py
-------------------
Clears ALL pothole + hazard + sign detection data so you can test fresh.
Keeps:  users, vehicles, driver_behavior, manual map pins (map_drop source)
Deletes: events_v2 rows of type pothole/hazard/sign
         potholes rows from video_detection / video source
"""
import sqlite3, os

BASE = os.path.dirname(os.path.abspath(__file__))

# 1. Clear events_v2 detection rows (potholes, hazards, signs, HIGH_RISK)
db1 = os.path.join(BASE, 'new.db')
conn = sqlite3.connect(db1)
c = conn.cursor()

c.execute("SELECT COUNT(*) FROM events_v2 WHERE type IN ('pothole','hazard','sign','HIGH_RISK')")
count = c.fetchone()[0]
print(f"events_v2 — detection rows to delete: {count}")

c.execute("DELETE FROM events_v2 WHERE type IN ('pothole','hazard','sign','HIGH_RISK')")
conn.commit()
conn.close()
print("  ✓ events_v2 detection rows cleared")

# 2. Clear video-detected potholes from potholes table
#    (keep source='map_drop' — these are manual map pins placed by the user)
db2 = os.path.join(BASE, 'database', 'safety_events.db')
conn2 = sqlite3.connect(db2)
c2 = conn2.cursor()

c2.execute("SELECT COUNT(*) FROM potholes WHERE source != 'map_drop'")
count2 = c2.fetchone()[0]
print(f"potholes — video-detected rows to delete: {count2}")

c2.execute("DELETE FROM potholes WHERE source != 'map_drop'")
conn2.commit()
conn2.close()
print("  ✓ video-detected potholes cleared (manual pins kept)")

# 3. Summary
conn = sqlite3.connect(db1)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM events_v2")
print(f"\nevents_v2 remaining rows: {c.fetchone()[0]}")
conn.close()

conn2 = sqlite3.connect(db2)
c2 = conn2.cursor()
c2.execute("SELECT COUNT(*) FROM potholes")
print(f"potholes remaining rows : {c2.fetchone()[0]}")
conn2.close()

print("\nDone. Now restart the API server and upload a test video.")
