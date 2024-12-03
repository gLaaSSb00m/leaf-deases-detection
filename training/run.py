import os
from ultralytics import YOLO
import cv2
import torch
import logging

# Enable CUDA launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.cuda.empty_cache()

def train_model():
    # Set the video path
    video_path = 'b.mp4'
    video_path_out = '{}_out.mp4'.format(video_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        logging.error("Could not read video file. Please check the path or video format.")
        cap.release()
        exit()

    H, W, _ = frame.shape

    # Initialize the video writer with 'mp4v' codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path_out, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    # Path to the model weights
    model_path = r'runs/detect/train/weights/best.pt'

    # Load the custom model and ensure it's on the GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    model = YOLO(model_path).to(device)

    threshold = 0.20

    # Process each frame
    while ret:
        results = model.predict(frame)

        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Extract confidence score and class
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                
                # Draw bounding boxes and labels if the score is above the threshold
                if confidence > threshold:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{model.names[class_id]}: {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('YOLOv8 Detection', frame)

        # Write the processed frame to the output file
        out.write(frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Read the next frame
        ret, frame = cap.read()

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    train_model()
