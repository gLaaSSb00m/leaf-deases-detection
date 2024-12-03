from flask import Flask, render_template, request
import torch

app = Flask(__name__)

# Load models
tea_model = torch.hub.load('ultralytics/yolov8', 'custom', path='path/to/tea/best.pt')
lemon_model = torch.hub.load('ultralytics/yolov8', 'custom', path='path/to/lemon/best.pt')
tomato_model = torch.hub.load('ultralytics/yolov8', 'custom', path='path/to/tomato/best.pt')

# Disease dictionaries
tea_leaf_diseases = {
    0: "Algal Leaf Spot",
    1: "Brown Blight",
    2: "Gray Blight",
    3: "Healthy",
    4: "Helopeltis",
    5: "Red Leaf Spot"
}

lemon_leaf_diseases = {
    0: "Canker"
}

tomato_leaf_diseases = {
    0: "Tomato Early Blight",
    1: "Tomato Bacterial Spot",
    2: "Tomato Late Blight",
    3: "Tomato Mold",
    4: "Tomato Septoria Leaf Spot"
}

# Function to detect disease
def detect_disease(leaf_type, image_path):
    if leaf_type == "tea":
        results = tea_model(image_path)
        disease_dict = tea_leaf_diseases
    elif leaf_type == "lemon":
        results = lemon_model(image_path)
        disease_dict = lemon_leaf_diseases
    elif leaf_type == "tomato":
        results = tomato_model(image_path)
        disease_dict = tomato_leaf_diseases
    else:
        return "Unknown leaf type", "N/A"

    for result in results:
        disease_class = result['class']
        disease_name = disease_dict.get(disease_class, "Unknown Disease")
        return leaf_type.capitalize(), disease_name

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/detect', methods=['POST'])
def detect():
    leaf_type = request.form.get('leaf_type')
    file = request.files['file']
    file_path = f'uploads/{file.filename}'
    file.save(file_path)
    leaf, disease = detect_disease(leaf_type, file_path)
    return render_template('templates/result.html', leaf=leaf, disease=disease)

if __name__ == '__main__':
    app.run(debug=True)
