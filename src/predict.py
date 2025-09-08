# src/predict.py
import joblib
import numpy as np

bundle = joblib.load("models/stress_model.joblib")
model = bundle["model"]
le = bundle["label_encoder"]
features = bundle["features"]

def predict_sample(sample_dict):
    X_row = np.array([[sample_dict[f] for f in features]], dtype=float)
    proba = model.predict_proba(X_row)[0]
    idx = proba.argmax()
    label = le.inverse_transform([idx])[0]
    return {"label": label, "probabilities": dict(zip(le.classes_, proba.tolist()))}

# example
if __name__ == "__main__":
    sample = {"sleep_hours":6, "work_hours":10, "exercise_hours":0.5, "screen_time":8, "water_intake":1.5, "mood_rating":3}
    print(predict_sample(sample))
