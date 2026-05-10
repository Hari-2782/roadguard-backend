from ultralytics import YOLO
import os
import shutil
import glob
import random
import yaml

# --- 1. Dataset Preparation ---
BASE_DIR = "custom_dataset"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(BASE_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val", "images")
VAL_LBL_DIR = os.path.join(BASE_DIR, "val", "labels")

# Create dirs
for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR]:
    os.makedirs(d, exist_ok=True)

# Source files
src_images = glob.glob(os.path.join(BASE_DIR, "images", "*.jpg"))
random.shuffle(src_images)

# Split 80/20
split_idx = int(len(src_images) * 0.8)
train_files = src_images[:split_idx]
val_files = src_images[split_idx:]

def move_files(files, dest_img, dest_lbl):
    for img_path in files:
        # Move Image
        base = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dest_img, base))
        
        # Move Label
        lbl_name = base.replace(".jpg", ".txt")
        lbl_path = os.path.join(BASE_DIR, "labels", lbl_name)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(dest_lbl, lbl_name))

print(f"Preparing Dataset: {len(train_files)} Train, {len(val_files)} Val")
move_files(train_files, TRAIN_IMG_DIR, TRAIN_LBL_DIR)
move_files(val_files, VAL_IMG_DIR, VAL_LBL_DIR)

# --- 2. Create YAML ---
yaml_content = {
    'path': os.path.abspath(BASE_DIR),
    'train': 'train/images',
    'val': 'val/images',
    'names': {
        0: "Speed-Limit-40-Kmph",
        1: "Speed-Limit-50-Kmph",
        2: "Speed-Limit-60-Kmph",
        3: "T-Junction-Ahead",
        4: "Cross-Roads-Ahead",
        5: "Roundabout-Ahead",
        6: "Traffic-from-left-merges-ahead",
        7: "Traffic-from-right-merges-ahead"
    }
}

yaml_path = os.path.join(BASE_DIR, "custom_data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f)

print(f"Dataset YAML created at {yaml_path}")

# --- 3. Train Model ---
if __name__ == '__main__':
    print("Starting Training...")
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        name='custom_domain_Model',
        exist_ok=True,
        workers=0  # Reduce workers for safety on Windows
    )

    print("Training Complete. Best weights saved to runs/detect/custom_domain_Model/weights/best.pt")
