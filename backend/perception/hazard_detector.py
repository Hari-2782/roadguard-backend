from ultralytics import YOLO
import logging
import os
import math

class HazardDetector:
    def __init__(self, model_path="models/hazard_model/best.pt"):
        self.logger = logging.getLogger("HazardDetector")
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                self.logger.info(f"Loaded Hazard YOLO model from {self.model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load hazard model: {e}")
        else:
            self.logger.warning("Hazard model not found. Hazard detection disabled.")

    def detect(self, frame):
        """
        Detect road hazards.
        """
        hazards = []
        if self.model is None:
            return hazards

        results = self.model(frame, imgsz=512, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = self.model.names[cls]
                
                # Estimate severity based on size relative to frame
                h_img, w_img = frame.shape[:2]
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = h_img * w_img
                ratio = box_area / frame_area
                
                # Simple heuristic: bigger = closer/more severe
                # Normalize 0.0 to 1.0 (clamped)
                severity = min(ratio * 10, 1.0) 

                hazards.append({
                    "type": "hazard",
                    "subtype": cls_name,
                    "severity": severity,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf
                })

        return hazards
