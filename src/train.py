# src/train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ------------------ Load dataset ------------------
df = pd.read_csv("data/stress_data.csv")

# Clean column names: remove spaces, lowercase, replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Debug: print all columns to verify
print("Columns in CSV:", df.columns.tolist())

# Ensure 'stress_level' exists
if "stress_level" not in df.columns:
    raise KeyError("'stress_level' column not found. Columns available:", df.columns.tolist())

# ------------------ Features & Target ------------------
X = df.drop("stress_level", axis=1)
y = df["stress_level"]

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)   # e.g., ['High','Low','Medium']

# ------------------ Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ------------------ Pipeline ------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# ------------------ Cross-validation ------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"acc": "accuracy", "f1_macro": "f1_macro"}
res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
print("CV accuracy:", round(res["test_acc"].mean(), 4), "f1_macro:", round(res["test_f1_macro"].mean(), 4))

# ------------------ Fit full training set ------------------
pipe.fit(X_train, y_train)

# ------------------ Evaluate ------------------
y_pred = pipe.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# ------------------ Confusion Matrix ------------------
os.makedirs("models", exist_ok=True)
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=le.classes_, cmap='Blues', normalize='true'
)
plt.title("Confusion Matrix (Normalized)")
plt.savefig("models/confusion_matrix.png", dpi=200, bbox_inches='tight')
plt.close()

# ------------------ Save Model ------------------
joblib.dump({"model": pipe, "label_encoder": le, "features": X.columns.tolist()}, "models/stress_model.joblib")
print("Saved models/stress_model.joblib successfully!")
