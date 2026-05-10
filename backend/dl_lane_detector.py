"""
Deep Learning Lane Detector
Replaces traditional CV-based lane detection with a trained neural network.

Usage:
    from dl_lane_detector import DLLaneDetector
    
    detector = DLLaneDetector("models/lane_model/lane_detector.pth")
    lane_info = detector.detect(frame)
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleLaneNet(nn.Module):
    """Simplified lane segmentation network - MUST match train_lane_model.py"""
    
    def __init__(self):
        super(SimpleLaneNet, self).__init__()
        
        # Encoder (3 levels only for stability)
        self.enc1 = self._conv_block(3, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        # Output: 3 channels (left, center, right lanes)
        self.out = nn.Conv2d(32, 3, 1)
        
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
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out(d1)
        return torch.sigmoid(out)


class DLLaneDetector:
    """Deep Learning based lane detector."""
    
    def __init__(self, model_path="models/lane_model/lane_detector.pth"):
        self.model_path = Path(model_path)
        self.model = None
        self.img_size = (320, 192)  # Must match training size (320x192)
        self.threshold = 0.5  # Confidence threshold
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            print(f"Warning: Lane model not found at {self.model_path}")
            print("Using fallback CV-based detection")
            return
        
        try:
            self.model = SimpleLaneNet().to(device)
            checkpoint = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"[DL Lane Detector] Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading lane model: {e}")
            self.model = None
    
    def detect(self, frame):
        """
        Detect lane lines in frame.
        
        Returns dict compatible with centerline.py format:
        {
            "left": {"x_bottom": int, "x_top": int, "type": str, "color": str},
            "right": {"x_bottom": int, "x_top": int, "type": str, "color": str}
        }
        """
        if self.model is None:
            # Fallback to traditional detection
            from centerline import detect_lane_lines
            return detect_lane_lines(frame)
        
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess
        img = cv2.resize(frame, self.img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Post-process
        self.last_masks = output[0].cpu().numpy()  # Store for drawing
        masks = self.last_masks
        
        # Extract lane lines from masks
        lane_info = {
            "left": {},       # Center line (white)
            "right": {},      # Right edge (not typically used in Sri Lanka)
            "yellow_edge": {} # Left yellow edge (channel 0)
        }
        
        # Channel 0 = left_edge (YELLOW edge on LEFT side of road)
        # Channel 1 = center_line (WHITE center line)
        # Channel 2 = right_edge (not typically present on Sri Lankan roads)
        
        # Process LEFT YELLOW EDGE (channel 0) - for yellow edge violation
        left_edge_mask = masks[0]
        left_edge_lane = self._extract_lane_from_mask(left_edge_mask, orig_w, orig_h)
        if left_edge_lane:
            lane_info["yellow_edge"] = {
                "x_bottom": left_edge_lane["x_bottom"],
                "x_top": left_edge_lane["x_top"],
                "type": "solid",
                "color": "yellow"
            }
        
        # Process CENTER WHITE LINE (channel 1) - for center line violation
        center_mask = masks[1]
        center_lane = self._extract_lane_from_mask(center_mask, orig_w, orig_h)
        if center_lane:
            lane_info["left"] = {
                "x_bottom": center_lane["x_bottom"],
                "x_top": center_lane["x_top"],
                "type": "solid",  # SOLID for violation detection
                "color": "white"
            }
        
        # Process RIGHT EDGE (channel 2) - optional, may not exist
        right_mask = masks[2]
        right_lane = self._extract_lane_from_mask(right_mask, orig_w, orig_h)
        if right_lane:
            lane_info["right"] = {
                "x_bottom": right_lane["x_bottom"],
                "x_top": right_lane["x_top"],
                "type": "solid",
                "color": "yellow"
            }
        
        return lane_info
    
    def _extract_lane_from_mask(self, mask, orig_w, orig_h):
        """Extract lane line coordinates from segmentation mask."""
        # Threshold mask
        binary = (mask > self.threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest) < 10:  # Too small
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)
        
        # Find bottom and top points of the lane
        points = largest.reshape(-1, 2)
        
        # Bottom point (highest y value)
        bottom_idx = np.argmax(points[:, 1])
        bottom_x = points[bottom_idx, 0]
        
        # Top point (lowest y value)
        top_idx = np.argmin(points[:, 1])
        top_x = points[top_idx, 0]
        
        # Scale back to original size
        scale_x = orig_w / self.img_size[0]
        scale_y = orig_h / self.img_size[1]
        
        return {
            "x_bottom": int(bottom_x * scale_x),
            "x_top": int(top_x * scale_x),
            "y_bottom": int(points[bottom_idx, 1] * scale_y),
            "y_top": int(points[top_idx, 1] * scale_y)
        }
    
    def draw_lanes(self, frame, lane_info):
        """Draw detected lanes on frame with colored overlay."""
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # If we have stored masks, draw colored overlay
        if hasattr(self, 'last_masks') and self.last_masks is not None:
            masks = self.last_masks
            
            # Create colored overlay from masks
            overlay = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            overlay[:, :, 2] = (masks[0] * 255).astype(np.uint8)  # Left edge - Red
            overlay[:, :, 1] = (masks[1] * 255).astype(np.uint8)  # Center line - Green
            overlay[:, :, 0] = (masks[2] * 255).astype(np.uint8)  # Right edge - Blue
            
            # Resize overlay to match frame
            overlay = cv2.resize(overlay, (w, h))
            
            # Blend with original frame
            result = cv2.addWeighted(result, 0.7, overlay, 0.5, 0)
        
        # Also draw lines for clarity
        if lane_info.get("left") and lane_info["left"].get("x_bottom"):
            left = lane_info["left"]
            x_bot = left["x_bottom"]
            x_top = left["x_top"]
            color = (0, 255, 0)  # Green for center line
            cv2.line(result, (x_bot, h), (x_top, int(h * 0.5)), color, 4)
        
        if lane_info.get("right") and lane_info["right"].get("x_bottom"):
            right = lane_info["right"]
            x_bot = right["x_bottom"]
            x_top = right["x_top"]
            color = (255, 0, 0)  # Blue for right edge
            cv2.line(result, (x_bot, h), (x_top, int(h * 0.5)), color, 4)
        
        return result


# Test function
if __name__ == "__main__":
    import sys
    
    detector = DLLaneDetector()
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else "Videos_test/WhiteLineCrossing.mp4"
    
    cap = cv2.VideoCapture(video_path)
    
    print("Testing DL Lane Detector")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect lanes
        lane_info = detector.detect(frame)
        
        # Draw
        result = detector.draw_lanes(frame, lane_info)
        
        # Resize for display
        h, w = result.shape[:2]
        if w > 1280:
            scale = 1280 / w
            result = cv2.resize(result, (int(w * scale), int(h * scale)))
        
        cv2.imshow("DL Lane Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
