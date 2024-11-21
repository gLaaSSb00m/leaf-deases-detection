import os
from ultralytics import YOLO
from roboflow import Roboflow
from ultralytics import YOLO

# Load the YOLOv8 model

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Define the dataset path
data_path = 'datasets/tea-leaf-diseases-6el9p-2/data.yaml'

# Train the model
results = model.train(
    data='datasets/tea-leaf-diseases-6el9p-2/data.yaml',
    epochs=50,
    imgsz=800,
    plots=True
)
