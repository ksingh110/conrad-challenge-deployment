from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = "lfc_5/penalized_lr_immunotherapy_model.pkl"
FEATURE_PATH = "lfc_5/feature_names.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURE_PATH, "rb") as f:
    feature_names = pickle.load(f)

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        return jsonify({"error": "CSV only for now"}), 400

    try:
        X = df[feature_names].astype(float)
    except KeyError as e:
        return jsonify({"error": f"Missing feature {str(e)}"}), 400

    prob = model.predict_proba(X)[0, 1]

    return jsonify({
        "probability": float(prob)
    })
