import torch
from ultralytics import YOLO
torch.cuda.empty_cache()

def train_segmentation_model():
    model = YOLO('yolov8n.pt').to('cuda')  # Ensure model is on GPU
    data_path = 'datasets2/Lemon-Disease-Detection-2/data.yaml'

    # Train the model with mixed precision and optimized DataLoader
    results = model.train(
        data=data_path,
        epochs=50,
        imgsz=800,
        plots=True,
        device=0,
        batch=4,  # Adjust based on your GPU memory
        
    )

if __name__ == '__main__':
    torch.cuda.empty_cache()  # Clear CUDA cache
    train_segmentation_model()
