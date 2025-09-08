import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load model
data = joblib.load("models/stress_model_tuned.joblib")
pipe = data["model"]
le = data["label_encoder"]
features = data["features"]

# Load CSV for test set
df = pd.read_csv("data/stress_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
target_column = [col for col in df.columns if "stress" in col][0]

X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode labels
y_enc = le.transform(y)

model = pipe.named_steps["clf"]

# Get sorted feature importances
fi = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
print("Feature importances:")
for feature, importance in fi:
    print(f"{feature}: {importance:.4f}")

# Plot feature importances
plt.figure(figsize=(10,6))
plt.barh([f[0] for f in fi[::-1]], [f[1] for f in fi[::-1]])
plt.xlabel("Importance")
plt.title("Feature Importances (RandomForest)")
plt.tight_layout()
plt.savefig("models/feature_importances.png", dpi=200)
plt.close()

# SHAP explainability
explainer = shap.TreeExplainer(pipe.named_steps['clf'])
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, feature_names=features, show=False)
plt.savefig("models/shap_summary.png", dpi=200, bbox_inches='tight')
plt.close()
