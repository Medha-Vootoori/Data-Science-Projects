# train_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_data, preprocess_data

# Paths
DATA_PATH = "data/bank_churn.csv"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and preprocess data
df = load_data(DATA_PATH)
df = preprocess_data(df)  # returns DataFrame

# Features and target
TARGET_COL = "churn"
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Save feature names
model_features = X.columns.tolist()
joblib.dump(model_features, os.path.join(MODEL_DIR, "model_features.pkl"))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save best model
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
print(f"✅ Best model saved with accuracy: {best_accuracy:.4f}")
