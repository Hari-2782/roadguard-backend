import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import os
import logging

try:
    from modules.utils.road_model import SimpleFastSCNN
except ImportError:
    try:
        from utils.road_model import SimpleFastSCNN
    except ImportError:
        SimpleFastSCNN = None

class RoadSegmenter:
    """
    Identifies drivable road area and filters detections from other models.
    """
    def __init__(self, model_path="models/road_segmentation/best.pth", device=None):
        self.logger = logging.getLogger("RoadSegmenter")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Road segmentation model not found at {self.model_path}. Running in fallback mode.")
            return

        if SimpleFastSCNN is None:
            self.logger.warning("SimpleFastSCNN class not available. Fallback mode.")
            return

        try:
            self.model = SimpleFastSCNN().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Handle both full model save and state_dict save
            if isinstance(checkpoint, dict):
                state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            elif isinstance(checkpoint, nn.Module):
                self.model = checkpoint.eval().to(self.device)
                self.logger.info(f"Loaded road segmentation model (full) from {self.model_path}")
                return
            else:
                state = checkpoint
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self.logger.info(f"Loaded road segmentation model from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load road model: {e}. Falling back to heuristic.")
            self.model = None

    def segment(self, frame):
        """
        Segment the drivable road area.
        Returns a binary mask (0-255) where 255 is road.
        """
        if self.model is None:
            # Fallback: Assume bottom 60% of screen is road if no model
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (0, int(h * 0.4)), (w, h), 255, -1)
            return mask

        try:
            # Preprocess
            img = cv2.resize(frame, (384, 384))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img)
                # Assuming output is [1, 1, H, W] or [1, 2, H, W]
                # Adjust based on model output
                if output.shape[1] > 1:
                    mask = torch.argmax(output, dim=1)
                else:
                    mask = (torch.sigmoid(output) > 0.5)
            
            mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            return mask
            
        except Exception as e:
            self.logger.error(f"Segmentation error: {e}")
            return np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255

    def filter_detections(self, detections, road_mask, threshold=0.5):
        """
        Filter out detections that are not significantly within the road mask.
        detections: list of dicts with 'bbox': [x1, y1, x2, y2]
        """
        filtered = []
        
        # Ensure mask is binary
        _, binary_mask = cv2.threshold(road_mask, 127, 255, cv2.THRESH_BINARY)
        
        road_area_mask = (binary_mask > 0)

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(road_mask.shape[1], x2)
            y2 = min(road_mask.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area <= 0:
                continue

            # Calculate intersection
            # We slice the mask at the bbox coordinates
            roi = road_area_mask[y1:y2, x1:x2]
            intersection_area = np.sum(roi)
            
            ratio = intersection_area / bbox_area
            
            if ratio >= threshold:
                filtered.append(det)
            else:
                self.logger.debug(f"Filtered detection {det.get('type')} - Overlap ratio: {ratio:.2f}")

        return filtered
