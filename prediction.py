
import json
import gzip
import joblib
import numpy as np

# Load trained model
with gzip.open("model.sav", "rb") as f:
    model = joblib.load(f)

# Load scaler
with gzip.open("scaler.sav", "rb") as f:
    scaler = joblib.load(f)

# Load feature order
with open("features.json", "r") as f:
    feature_order = json.load(f)

# Load threshold
with open("threshold.txt", "r") as f:
    threshold = float(f.read().strip())


def predict_diabetes(user_input_dict):
    values = [user_input_dict[f] for f in feature_order]
    X = np.array([values], dtype=float)

    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]

    pred = int(prob >= threshold)
    label = "Diabetes" if pred == 1 else "Tidak Diabetes"

    return label, float(prob)
