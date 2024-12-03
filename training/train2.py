import os
from ultralytics import YOLO

from ultralytics import YOLO

# Load the YOLOv8 model

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Define the dataset path
data_path = 'datasets/tea-leaf-diseases-6el9p-2/data.yaml'

# Train the model
results = model.train(
    data='datasets/tea-leaf-diseases-6el9p-2/data.yaml',
    epochs=50,
    imgsz=800,
    plots=True
)



# Load YOLO model
model = YOLO("yolov8n.pt")  # Choose a base model (n, s, m, l, x)

# Train the model
model.train(
    data="data.yaml",  # Path to your data config
    epochs=100,        # Number of epochs
    imgsz=640,         # Image size
    device=0           # Use GPU (set to 'cpu' for CPU)
)
