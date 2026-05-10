from ultralytics import YOLO
import numpy as np
import logging
import os

class PotholeDetector:
    """
    Pothole detector with a spatial Kalman/EMA confidence map to suppress
    single-frame false positives.  Each frame the map decays; detections
    accumulate confidence until they surpass CONFIRM_THRESHOLD, at which
    point they are returned as confirmed potholes.
    """

    # Tuning knobs
    _GRID        = 8      # divide frame into GRID x GRID spatial cells
    _EMA_ALPHA   = 0.55   # how quickly confidence builds  (0=never, 1=instant)
    _DECAY       = 0.80   # per-frame decay of existing confidence
    _CONFIRM_TH  = 0.45   # minimum accumulated confidence to report

    def __init__(self, model_path="models/pothole_model/best copy.pt"):
        self.logger = logging.getLogger("PotholeDetector")
        self.model_path = model_path
        self.model = None
        self._conf_map = None   # lazily initialised per first frame size
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                self.logger.info(f"Loaded Pothole YOLO model from {self.model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load pothole model: {e}")
        else:
            self.logger.warning("Pothole model not found. Pothole detection disabled.")

    def reset(self):
        """Clear accumulated confidence (call when switching video sources)."""
        self._conf_map = None

    def detect(self, frame):
        potholes = []
        if self.model is None:
            return potholes

        h_img, w_img = frame.shape[:2]

        # Lazily init or re-init confidence map when frame size changes
        if self._conf_map is None or self._conf_map.shape != (self._GRID, self._GRID):
            self._conf_map = np.zeros((self._GRID, self._GRID), dtype=np.float32)

        # ── Step 1: decay the entire map each frame ──────────────────────
        self._conf_map *= self._DECAY

        cell_h = h_img / self._GRID
        cell_w = w_img / self._GRID

        results = self.model(frame, imgsz=512, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                # Severity calculation (same as before)
                box_h   = y2 - y1
                center_y = (y1 + y2) / 2
                center_x = (x1 + x2) / 2

                pos_factor  = center_y / h_img
                size_factor = min((box_h / h_img) * 5, 1.0)
                severity    = min(max(pos_factor * 0.6 + size_factor * 0.4, 0.0), 1.0)

                # Map detection to grid cell
                row = int(min(center_y / cell_h, self._GRID - 1))
                col = int(min(center_x / cell_w, self._GRID - 1))

                # ── Step 2: EMA accumulate confidence in that cell ────────
                self._conf_map[row, col] = (
                    self._EMA_ALPHA * conf
                    + (1 - self._EMA_ALPHA) * self._conf_map[row, col]
                )

                # ── Step 3: only report if accumulated confidence is high ─
                if self._conf_map[row, col] >= self._CONFIRM_TH:
                    potholes.append({
                        "type":       "pothole",
                        "severity":   float(severity),
                        "confidence": float(self._conf_map[row, col]),
                        "bbox":       [float(x1), float(y1), float(x2), float(y2)],
                    })

        return potholes

