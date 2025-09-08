# src/train_step5.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load dataset
# ---------------------------
csv_path = "data/stress_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

df = pd.read_csv(csv_path)
print("Original CSV columns:", df.columns.tolist())

# ---------------------------
# 2. Normalize column names
# ---------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Normalized columns:", df.columns.tolist())

# ---------------------------
# 3. Detect target column
# ---------------------------
# Try to find a column containing 'stress'
possible_targets = [col for col in df.columns if "stress" in col]
if not possible_targets:
    raise KeyError("No target column containing 'stress' found in CSV!")
target_column = possible_targets[0]
print(f"Using target column: '{target_column}'")

# ---------------------------
# 4. Prepare features & target
# ---------------------------
X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode target labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ---------------------------
# 5. Train/Test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ---------------------------
# 6. Define cross-validation
# ---------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------
# 7. Define pipeline
# ---------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

# ---------------------------
# 8. Hyperparameter tuning
# ---------------------------
param_dist = {
    "clf__n_estimators": [100, 200, 400, 800],
    "clf__max_depth": [None, 6, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4]
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=12,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("Best CV f1_macro:", search.best_score_)

best_pipe = search.best_estimator_

# ---------------------------
# 9. Evaluate on test set
# ---------------------------
y_pred = best_pipe.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save confusion matrix
os.makedirs("models", exist_ok=True)
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=le.classes_, cmap='Blues', normalize='true'
)
plt.title("Confusion Matrix (Tuned RandomForest)")
plt.savefig("models/confusion_matrix_tuned.png", dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------
# 10. Save model
# ---------------------------
model_path = "models/stress_model_tuned.joblib"
joblib.dump({"model": best_pipe, "label_encoder": le, "features": X.columns.tolist()}, model_path)
print(f"Saved {model_path} successfully!")
