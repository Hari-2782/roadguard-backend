"""
Lane Detection Model Training Script
Trains a lane detection model using annotated data.

Uses a simple but effective approach:
1. Semantic segmentation for lane pixels
2. Post-processing to extract lane lines
"""
import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LaneDataset(Dataset):
    """Dataset for lane detection training."""
    
    def __init__(self, image_dir, annotation_dir, img_size=(320, 192)):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.img_size = img_size  # (width, height) - must be divisible by 16
        
        # Get all annotated images (exclude progress.json and other non-annotation files)
        all_json = list(self.annotation_dir.glob("*.json"))
        self.annotations = [f for f in all_json if f.name != "progress.json"]
        print(f"Found {len(self.annotations)} annotated images")
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Load annotation
        ann_path = self.annotations[idx]
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        
        # Load image
        img_path = self.image_dir / ann['image']
        img = cv2.imread(str(img_path))
        if img is None:
            # Return zeros if image not found
            return torch.zeros(3, self.img_size[1], self.img_size[0]), torch.zeros(3, self.img_size[1], self.img_size[0])
        
        orig_h, orig_w = img.shape[:2]
        
        # Resize image
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        # Create lane mask (3 channels: left, center, right)
        mask = np.zeros((3, self.img_size[1], self.img_size[0]), dtype=np.float32)
        
        lane_channels = {'left_edge': 0, 'center_line': 1, 'right_edge': 2}
        
        for lane_name, channel in lane_channels.items():
            lane_data = ann['lanes'].get(lane_name, {})
            points = lane_data.get('points', [])
            
            if len(points) >= 2:
                # Scale points to new size
                scaled_points = []
                for x, y in points:
                    new_x = int(x * self.img_size[0] / orig_w)
                    new_y = int(y * self.img_size[1] / orig_h)
                    scaled_points.append([new_x, new_y])
                
                # Draw lane line on a 2D temp mask first
                pts = np.array(scaled_points, dtype=np.int32)
                temp_mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)
                cv2.polylines(temp_mask, [pts], False, 255, thickness=8)
                mask[channel] = temp_mask.astype(np.float32) / 255.0
        
        return torch.from_numpy(img), torch.from_numpy(mask)


class SimpleLaneNet(nn.Module):
    """Simplified lane segmentation network."""
    
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
        # Store original size for final resize
        orig_size = x.shape[2:]  # (H, W)
        
        # Encoder
        e1 = self.enc1(x)        # 320x192 -> 320x192
        e2 = self.enc2(self.pool(e1))  # -> 160x96
        e3 = self.enc3(self.pool(e2))  # -> 80x48
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # -> 40x24
        
        # Decoder with skip connections
        d3 = self.up3(b)  # -> 80x48
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)  # -> 160x96
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)  # -> 320x192
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out(d1)
        return torch.sigmoid(out)


def train_model(epochs=50, batch_size=4, lr=0.001):
    """Train the lane detection model."""
    
    image_dir = "lane_dataset/images"
    annotation_dir = "lane_dataset/annotations"
    model_save_path = "models/lane_detector.pth"
    
    # Create model directory
    os.makedirs("models", exist_ok=True)
    
    # Check annotations exist
    ann_files = list(Path(annotation_dir).glob("*.json"))
    if len(ann_files) < 5:
        print(f"Error: Only {len(ann_files)} annotations found.")
        print("Please annotate at least 20-30 images first!")
        print("Run: python lane_annotator.py")
        return
    
    print(f"\nFound {len(ann_files)} annotated images")
    
    # Create dataset and dataloader
    dataset = LaneDataset(image_dir, annotation_dir)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create model
    model = SimpleLaneNet().to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_save_path)
    
    print("\n" + "=" * 50)
    print(f"Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_save_path}")
    print("=" * 50)
    
    return model


def test_model(video_path=None):
    """Test the trained model on a video or camera."""
    
    model_path = "models/lane_detector.pth"
    
    if not os.path.exists(model_path):
        print("Error: No trained model found!")
        print("Run training first: python train_lane_model.py")
        return
    
    # Load model
    model = SimpleLaneNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Open video
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess (must match training size: 320x192)
        img = cv2.resize(frame, (320, 192))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
        
        # Post-process
        mask = output[0].cpu().numpy()
        
        # Create colored overlay
        overlay = np.zeros_like(img)
        overlay[:, :, 2] = (mask[0] * 255).astype(np.uint8)  # Left - Red
        overlay[:, :, 1] = (mask[1] * 255).astype(np.uint8)  # Center - Green
        overlay[:, :, 0] = (mask[2] * 255).astype(np.uint8)  # Right - Blue
        
        # Resize back and blend
        overlay = cv2.resize(overlay, (orig_w, orig_h))
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Resize for display if too large
        display_h, display_w = result.shape[:2]
        if display_w > 1280:
            scale = 1280 / display_w
            result = cv2.resize(result, (int(display_w * scale), int(display_h * scale)))
        
        cv2.imshow("Lane Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        video = sys.argv[2] if len(sys.argv) > 2 else None
        test_model(video)
    else:
        train_model(epochs=50, batch_size=4)
