import sqlite3
import time
import os

class PotholeMapLogger:
    def __init__(self, db_path="database/safety_events.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS potholes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL,
                longitude REAL,
                severity REAL,
                timestamp REAL
            )
        """)
        conn.commit()
        conn.close()

    def log_pothole(self, lat, lon, severity):
        """
        Log pothole if not already logged recently nearby.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Simple duplicate check (very rough proximity)
        # In real world, use Haversine distance
        cur.execute("""
            SELECT id FROM potholes 
            WHERE abs(latitude - ?) < 0.0001 
            AND abs(longitude - ?) < 0.0001
        """, (lat, lon))
        
        if cur.fetchone() is None:
            cur.execute("""
                INSERT INTO potholes (latitude, longitude, severity, timestamp)
                VALUES (?, ?, ?, ?)
            """, (lat, lon, severity, time.time()))
            conn.commit()
            
        conn.close()
