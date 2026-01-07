from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Relative paths (Vercel-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "penalized_lr_immunotherapy_model_lfc_5.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_names_lfc5.pkl")

# Load model + features once (cold start)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURE_PATH, "rb") as f:
    feature_names = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Load data
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.filename.endswith(".txt"):
        df = pd.read_csv(file, sep="\t")
    elif file.filename.endswith(".json"):
        df = pd.read_json(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    # Validate features
    try:
        X = df[feature_names].astype(float)
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    if X.shape[1] != len(feature_names):
        return jsonify({"error": f"Expected {len(feature_names)} features"}), 400

    prob = model.predict_proba(X)[0, 1]

    return jsonify({"probability": float(prob)})
##sup
