from ultralytics import YOLO
from .model_config import models_info

def detect_leaf_and_disease(leaf_type, image_path):
    if leaf_type not in models_info:
        return {"error": f"Invalid leaf type: {leaf_type}"}

    # Load model and classes
    model_info = models_info[leaf_type]
    model = YOLO(model_info["model_path"])
    classes = model_info["classes"]

    # Perform detection
    results = model(image_path, conf=0.5)
    for result in results:
        if len(result.boxes) > 0:
            diseases = []
            for box in result.boxes:
                class_id = int(box.cls)
                disease_name = classes[class_id] if class_id < len(classes) else "Unknown"
                diseases.append(disease_name)
            return {"leaf": leaf_type.capitalize(), "diseases": diseases}

    return {"leaf": leaf_type.capitalize(), "diseases": ["No disease detected."]}
