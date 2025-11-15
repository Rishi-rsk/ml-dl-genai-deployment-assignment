from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return {"message": "ML Model Running! Use /predict"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]
    arr = np.array(data).reshape(1, -1)
    result = model.predict(arr).tolist()
    return {"prediction": result}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
