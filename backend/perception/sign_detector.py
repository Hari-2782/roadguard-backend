from ultralytics import YOLO
import cv2
from collections import deque, Counter
import time
import logging

class SignState:
    """
    Holds stable speed limit + stable junction voting + cooldown logic.
    """
    def __init__(self):
        self.speed_votes = deque(maxlen=15)
        self.current_speed_limit = None
        self.junction_votes = deque(maxlen=12)
        
    def update(self, detections):
        # Filter for speed signs
        speeds = [d for d in detections if "Speed-Limit" in d['subtype']]
        if speeds:
            # Extract number from "Speed-Limit-40-Kmph"
            val_str = speeds[0]['subtype'].split('-')[2]
            try:
                val = int(val_str)
                self.speed_votes.append(val)
            except:
                pass
        
        # Voting logic
        if len(self.speed_votes) >= 3:
            # Check last 3 are same
            last3 = list(self.speed_votes)[-3:]
            if len(set(last3)) == 1:
                self.current_speed_limit = last3[0]

    def get_current_limit(self):
        return self.current_speed_limit


class SignDetector:
    def __init__(self, model_path="models/sign_model/bestS.pt"):
        self.logger = logging.getLogger("SignDetector")
        self.model_path = model_path
        self.model = None
        self.state = SignState()
        self._load_model()

    def _load_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"Loaded Sign model from {self.model_path}")
        except Exception as e:
            self.logger.warning(f"Sign model not found at {self.model_path}: {e}")

    def detect(self, frame):
        if self.model is None:
            return []

        results = self.model(frame, imgsz=512, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                name = self.model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                detections.append({
                    "type": "sign",
                    "subtype": name,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf
                })
        
        # Update state for temporal consistency
        self.state.update(detections)
        
        return detections

    def get_active_speed_limit(self):
        return self.state.get_current_limit()
