from ultralytics import YOLO
import cv2
import time

class PotholeDetector:
    def __init__(self, model_path="pothole_model.pt", conf=0.4):
        self.model = None
        self.enabled = False
        try:
            self.model = YOLO(model_path)
            self.enabled = True
            print(f"[INFO] Pothole Detector loaded: {model_path}")
        except Exception as e:
            print(f"[WARNING] Pothole model not found at {model_path}: {e}")
            
        self.conf = conf
        # Track potholes to avoid duplicates (simple distance checks could be added here)
        self.detections = [] 

    def detect(self, frame):
        if not self.enabled:
            return []
            
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        return results[0]

    def draw(self, frame, results):
        if not self.enabled or results is None:
            return frame
            
        out_frame = frame.copy()
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls = int(box.cls.item())
            
            # Draw Red/Orange box for potholes
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(out_frame, f"Pothole {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        return out_frame
