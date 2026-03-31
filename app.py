import os
import re
import io

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import joblib

app = Flask(__name__)

# --------------- Load Models ---------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) Motorcycle bigbike classifier
bike_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model", "bigbike_high_precision_v2.keras")
)
bike_class_names = ["big_bike_500cc", "small_bike"]  # must match training order

# 2) Gambling text detector (sklearn pipeline inside joblib)
gambling_model = joblib.load(
    os.path.join(BASE_DIR, "model", "ensemble_gambling_model_v2.joblib")
)


# --------------- Helpers ---------------
def clean_text(text: str) -> str:
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', text)
    return text.strip()


# --------------- Routes ---------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/bike", methods=["POST"])
def predict_bike():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = bike_model.predict(img_array)
        result_idx = int(np.argmax(prediction))
        result_label = bike_class_names[result_idx]
        confidence = float(np.max(prediction))

        status = "ALLOW" if result_label == "big_bike_500cc" else "DENY"

        return jsonify(
            {
                "status": status,
                "class": result_label,
                "confidence": f"{confidence * 100:.2f}%",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/gambling", methods=["POST"])
def predict_gambling():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    raw_text = data["text"]
    if not raw_text.strip():
        return jsonify({"error": "Text is empty"}), 400

    try:
        prediction = gambling_model.predict([raw_text.strip()])[0]
        result = "Gambling" if prediction == 1 else "Clean"
        return jsonify({"result": result, "text": raw_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
