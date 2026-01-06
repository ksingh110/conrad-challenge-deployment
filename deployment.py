from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

MODEL_PATH = "/Users/krishaysingh/Documents/Holland_Lab_KidneyCancer/fred-hutch-immunotherapy-code/lfc_5/penalized_lr_immunotherapy_model_lfc_5.pkl"
FEATURE_PATH = "/Users/krishaysingh/Documents/Holland_Lab_KidneyCancer/fred-hutch-immunotherapy-code/lfc_5/feature_names_lfc5.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
feature_names = pickle.load(open(FEATURE_PATH, "rb"))
from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    file = request.files["file"]

    # Load uploaded data
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.filename.endswith(".txt"):
        df = pd.read_csv(file, sep="\t")
    elif file.filename.endswith(".json"):
        df = pd.read_json(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    # Enforce correct feature order
    try:
        X = df[feature_names].values.astype(float)
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    if X.shape[1] != feature_names.__len__():
        return jsonify({"error": f"Expected {feature_names.__len__()} features"}), 400

    X = df[feature_names].astype(float)

    prob = model.predict_proba(X)[0, 1]
    print("Raw probability:", prob)


    return jsonify({
        "probability": float(prob)
    })
if __name__ == "__main__":
    app.run(debug=True)
@app.route("/")
def index():
    return "FLASK IS WORKING"
