from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import pickle
from tensorflow import keras

app = Flask(__name__, static_folder='statics')

# ---------- Load model + preprocessing ----------
print("Loading cancer model...")
model = keras.models.load_model("cancer_model.keras")
print("Model loaded!")

print("Loading scaler + metadata...")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("metadata.json", "r") as f:
    meta = json.load(f)

FEATURES = meta["feature_names"]            # ordered list of 30 feature names
CLASSES  = meta["class_names"]              # ["malignant", "benign"]
# For HTML form names (no spaces), we’ll expose sanitized keys
SANITIZED_KEYS = [fn.replace(" ", "_") for fn in FEATURES]

print("Ready to predict.")
print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Classes: {CLASSES}")

@app.route("/")
def home():
    return "Welcome to the Breast Cancer Diagnosis API!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect inputs in the exact feature order
            values = []
            for fn in FEATURES:
                key = fn.replace(" ", "_")       # HTML field name
                if key not in request.form:
                    raise ValueError(f"Missing input: {key}")
                values.append(float(request.form[key]))

            X = np.array([values], dtype=np.float32)
            Xs = scaler.transform(X)

            # Model outputs P(benign) since label=1 is benign in sklearn dataset
            p_benign = float(model.predict(Xs, verbose=0)[0][0])
            p_malignant = 1.0 - p_benign

            pred_idx = int(p_benign >= 0.5)      # 0=malignant, 1=benign
            diagnosis = CLASSES[pred_idx]
            confidence = p_benign if diagnosis == "benign" else p_malignant

            return jsonify({
                "diagnosis": diagnosis,
                "confidence": confidence,
                "probabilities": {
                    "malignant": p_malignant,
                    "benign": p_benign
                }
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # GET → render the HTML form
    return render_template("predict.html",
                           features=FEATURES,
                           sanitized_keys=SANITIZED_KEYS,
                           class_names=CLASSES)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4000)
