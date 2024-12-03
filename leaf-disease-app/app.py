from flask import Flask, render_template, request, redirect, url_for
from models.detect import detect_leaf_and_disease
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create the uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the selected leaf type
        leaf_type = request.form.get("leaf_type")

        # Get the uploaded file
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("index.html", error="No file uploaded or invalid file.")

        file = request.files["image"]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Perform disease detection
        result = detect_leaf_and_disease(leaf_type, filepath)

        return render_template("index.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
