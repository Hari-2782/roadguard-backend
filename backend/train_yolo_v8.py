from ultralytics import YOLO
import torch

def train_model():
    # Check for GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # mosaic=1.0 is CRITICAL to force the model to learn from "collages" of images,
    # helping it understand that signs are objects IN a scene, not the whole scene.
    results = model.train(
        data='custom_dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        mosaic=1.0,
        mixup=0.1,
        device=device,
        project='runs/detect',
        name='train_v8_large',
        exist_ok=True
    )
    
    print("Training finished.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_model()
