from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
import os

app = Flask(__name__)

MODEL_PATH = "cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Place cnn_model.h5 in this folder.")

model = keras.models.load_model(MODEL_PATH)

@app.route("/")
def index():
    return {"message": "DL CNN Model Running! Use /predict with JSON { \"input\": [784 floats] }"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("input", None)
    if data is None:
        return jsonify({"error": "No 'input' key found. Provide flattened 28x28 array."}), 400

    arr = np.array(data, dtype=np.float32)

    if arr.size != 28*28:
        return jsonify({"error": f"Input size must be 784 values (received {arr.size})"}), 400

    arr = arr.reshape(1, 28, 28, 1)

    preds = model.predict(arr)
    label = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
