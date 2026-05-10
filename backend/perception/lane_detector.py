import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleLaneNet(nn.Module):
    """Simplified lane segmentation network"""
    def __init__(self):
        super(SimpleLaneNet, self).__init__()
        self.enc1 = self._conv_block(3, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.bottleneck = self._conv_block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        self.out = nn.Conv2d(32, 3, 1) # 3 channels: left, center, right
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.out(d1))

class LaneDetector:
    def __init__(self, model_path="models/lane_model/lane_detector.pth"):
        self.logger = logging.getLogger("LaneDetector")
        self.model_path = Path(model_path)
        self.model = None
        self.img_size = (320, 192)
        self._last_masks = None   # cached from most recent detect() call
        self._last_frame_size = None
        self._load_model()
        
    def _load_model(self):
        if not self.model_path.exists():
            self.logger.warning(f"Lane model not found at {self.model_path}")
            return
            
        try:
            self.model = SimpleLaneNet().to(device)
            # Handle loading state dict safely
            try:
                checkpoint = torch.load(self.model_path, map_location=device, weights_only=True)
            except:
                checkpoint = torch.load(self.model_path, map_location=device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            self.logger.info("Deep Learning Lane Model Loaded")
        except Exception as e:
            self.logger.error(f"Error loading lane model: {e}")

    def detect(self, frame):
        """
        Returns dictionary of lane info + violation status.
        Detects left lane (ch0), center line (ch1), right lane (ch2).
        Positions are scaled back to the original frame dimensions.
        """
        lane_status = {
            "deviation": 0.0,
            "is_crossing": False,
            "left_x": None,
            "right_x": None
        }

        if self.model is None:
            return lane_status

        frame_h, frame_w = frame.shape[:2]
        # self.img_size is (width, height) for cv2.resize
        model_w, model_h = self.img_size

        img = cv2.resize(frame, self.img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(img_tensor)

        masks = output[0].cpu().numpy()  # (3, model_h, model_w)
        self._last_masks = masks          # cache for draw_lanes()
        self._last_frame_size = (frame_w, frame_h)

        # Use the lower 40% of the model image for lane detection
        bottom_start = int(model_h * 0.6)

        # Channel 0 = left lane, Channel 1 = center line, Channel 2 = right lane
        left_mask   = masks[0]
        center_mask = masks[1]
        right_mask  = masks[2]

        # --- Left lane ---
        ys_l, xs_l = np.where(left_mask[bottom_start:, :] > 0.5)
        if len(xs_l) > 0:
            lane_status["left_x"] = float(np.mean(xs_l)) * frame_w / model_w

        # --- Right lane ---
        ys_r, xs_r = np.where(right_mask[bottom_start:, :] > 0.5)
        if len(xs_r) > 0:
            lane_status["right_x"] = float(np.mean(xs_r)) * frame_w / model_w

        # --- Fallback: use center-line channel if both sides missing ---
        if lane_status["left_x"] is None and lane_status["right_x"] is None:
            ys_c, xs_c = np.where(center_mask[bottom_start:, :] > 0.5)
            if len(xs_c) > 0:
                center_x = float(np.mean(xs_c)) * frame_w / model_w
                deviation = (frame_w / 2 - center_x) / frame_w
                lane_status["deviation"] = float(deviation)
                if abs(deviation) > 0.15:
                    lane_status["is_crossing"] = True
            return lane_status

        # --- Compute deviation from the mid-point of detected lanes ---
        left_x  = lane_status["left_x"]
        right_x = lane_status["right_x"]
        car_center = frame_w / 2

        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2
            deviation = (car_center - lane_center) / frame_w
        elif left_x is not None:
            deviation = (car_center - left_x) / frame_w
        else:
            deviation = (right_x - car_center) / frame_w

        lane_status["deviation"] = float(deviation)
        if abs(deviation) > 0.15:
            lane_status["is_crossing"] = True

        return lane_status

    def draw_lanes(self, frame, lane_info=None, is_violation=False):
        """
        Paint colored lane masks onto *frame* using the masks cached by
        the most recent detect() call, then return the annotated frame.

        Colors (BGR):
                    left lane   → red    (0, 0, 255)
                    center line → green  (0, 255, 0)
                    right lane  → blue   (255, 0, 0)
        """
        if self._last_masks is None:
            return frame   # model not loaded or detect() not yet called

        frame_h, frame_w = frame.shape[:2]
        model_w, model_h = self.img_size

        # Build a color mask at model resolution then resize to frame size
        color_mask_model = np.zeros((model_h, model_w, 3), dtype=np.uint8)

        left_mask   = self._last_masks[0]
        center_mask = self._last_masks[1]
        right_mask  = self._last_masks[2]

        color_mask_model[:, :, 2] = (left_mask * 255).astype(np.uint8)
        color_mask_model[:, :, 1] = (center_mask * 255).astype(np.uint8)
        color_mask_model[:, :, 0] = (right_mask * 255).astype(np.uint8)

        # Scale up to original frame size
        color_mask = cv2.resize(color_mask_model, (frame_w, frame_h),
                                interpolation=cv2.INTER_NEAREST)

        annotated = cv2.addWeighted(frame, 1.0, color_mask, 0.4, 0)
        return annotated
