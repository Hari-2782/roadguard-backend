from ultralytics import YOLO
import cv2
from collections import deque, Counter
import time

# --------- Class name -> speed value mapping ----------
# These match the 14-class focused model (speed_junction_v2)
SPEED_VALUE = {
    "Speed-15": 15,
    "Speed-30": 30,
    "Speed-40": 40,
    "Speed-50": 50,
    "Speed-60": 60,
    "Speed-70": 70,
    "Speed-80": 80,
    "Speed-100": 100,
}

JUNCTION_CLASSES = {
    "CrossRoads-Ahead",
    "T-Junction-Ahead",
    "Staggered-Junction",
    "Y-Junction-Ahead",
    "Traffic-Merge-Right",
    "Traffic-Merge-Left",
}

JUNCTION_SPEECH = {
    "CrossRoads-Ahead": "Cross roads ahead",
    "T-Junction-Ahead": "T junction ahead",
    "Staggered-Junction": "Staggered junction ahead",
    "Y-Junction-Ahead": "Y junction ahead",
    "Traffic-Merge-Right": "Traffic merges from right ahead",
    "Traffic-Merge-Left": "Traffic merges from left ahead",
}


class SignState:
    """
    Holds stable speed limit + stable junction voting + cooldown logic (anti-flicker).
    """
    def __init__(
        self,
        speed_vote_window=15,
        speed_min_hits=6,
        speed_hold_seconds=8.0,
        junction_vote_window=12,
        junction_min_hits=5,
        junction_cooldown_sec=6.0,
    ):
        # Speed stability
        self.speed_votes = deque(maxlen=speed_vote_window)
        self.speed_min_hits = speed_min_hits
        self.speed_hold_seconds = speed_hold_seconds
        self.current_speed_limit = None
        self.last_speed_update_time = 0.0

        # Junction stability
        self.junction_votes = deque(maxlen=junction_vote_window)
        self.junction_min_hits = junction_min_hits
        self.last_junction_alert_time = 0.0
        self.junction_cooldown_sec = junction_cooldown_sec

    def update_speed_limit(self, speed_value: int) -> bool:
        now = time.time()
        self.speed_votes.append(speed_value)

        counts = Counter(self.speed_votes)
        print(f"[DEBUG-VOTE] Buffer: {dict(counts)} Best: {counts.most_common(1)[0] if counts else 'None'} MinHits:{self.speed_min_hits}")
        
        if not counts:
            return False
            
        best_val, best_count = counts.most_common(1)[0]
        
        if best_count >= self.speed_min_hits and self.current_speed_limit != best_val:
            self.current_speed_limit = best_val
            self.last_speed_update_time = now
            return True
        
        return False


    def get_current_speed_limit(self):
        """
        Returns the confirmed speed limit. 
        Does NOT expire. It stays valid until the next sign is detected.
        """
        return self.current_speed_limit

    def update_junction_vote(self, cls_name: str):
        """
        Push a junction detection to the vote window.
        """
        self.junction_votes.append(cls_name)

    def pop_stable_junction(self):
        """
        Returns stable junction name if enough hits inside window, else None.
        """
        if not self.junction_votes:
            return None
        counts = Counter(self.junction_votes)
        best_val, best_count = counts.most_common(1)[0]
        if best_count >= self.junction_min_hits:
            return best_val
        return None

    def can_alert_junction(self) -> bool:
        now = time.time()
        if (now - self.last_junction_alert_time) >= self.junction_cooldown_sec:
            self.last_junction_alert_time = now
            return True
        return False


class YOLOSignDetector:
    def __init__(self, weights_path, conf=0.45, imgsz=640):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        """
        Returns list of detections:
        [
          { 'cls_name': str, 'conf': float, 'xyxy': (x1,y1,x2,y2) }
        ]
        """
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        r = results[0]

        dets = []
        if r.boxes is None:
            return dets

        names = r.names
        for b in r.boxes:
            cls_id = int(b.cls.item())
            cls_name = names[cls_id]
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            dets.append({"cls_name": cls_name, "conf": conf, "xyxy": (x1, y1, x2, y2)})
        return dets

    @staticmethod
    def draw(frame, dets):
        out = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            
            label = d["cls_name"]
            
            # Speed signs: Show "40 KM/H" format
            if label in SPEED_VALUE:
                speed_val = SPEED_VALUE[label]
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{speed_val} KM/H"
            # Junctions: Show "JUNCTION AHEAD"
            elif label in JUNCTION_CLASSES:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = "JUNCTION AHEAD"
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), (128, 128, 128), 2)
                
            cv2.putText(
                out,
                f'{label} {d["conf"]:.2f}',
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return out
