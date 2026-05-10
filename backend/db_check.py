import sqlite3, os

# Check safety_events.db (potholes)
db1 = 'database/safety_events.db'
print(f"\n=== {db1} ===")
if os.path.exists(db1):
    conn = sqlite3.connect(db1)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    print("Tables:", tables)
    for t in tables:
        c.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in c.fetchall()]
        print(f"  {t} columns: {cols}")
        c.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t} count: {c.fetchone()[0]}")
        c.execute(f"SELECT * FROM {t} LIMIT 3")
        rows = c.fetchall()
        for row in rows:
            print(f"    row: {dict(row)}")
    conn.close()
else:
    print("NOT FOUND")

# Check new.db (events_v2)
db2 = 'new.db'
print(f"\n=== {db2} ===")
if os.path.exists(db2):
    conn = sqlite3.connect(db2)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    print("Tables:", tables)
    for t in tables:
        c.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in c.fetchall()]
        print(f"  {t} columns: {cols}")
        c.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t} count: {c.fetchone()[0]}")
        # show pothole events specifically
        if t == 'events_v2':
            c.execute("SELECT DISTINCT type FROM events_v2")
            print(f"  distinct types: {[r[0] for r in c.fetchall()]}")
            c.execute("SELECT COUNT(*) FROM events_v2 WHERE type='pothole'")
            print(f"  pothole events: {c.fetchone()[0]}")
            c.execute("SELECT * FROM events_v2 WHERE type='pothole' LIMIT 3")
            rows = c.fetchall()
            for row in rows:
                print(f"    row: {dict(row)}")
    conn.close()
else:
    print("NOT FOUND")
