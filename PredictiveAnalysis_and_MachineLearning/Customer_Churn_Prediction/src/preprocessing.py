# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

DATA_PATH = "data/bank_churn.csv"
PROCESSED_DIR = "models/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    """Load dataset"""
    return pd.read_csv(path)

def preprocess_data(df):
    """Encode categorical and scale numerical"""
    df = df.copy()
    categorical_cols = ["country", "gender"]  # adjust your categorical columns
    label_encoders = {}

    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Save label encoders
    joblib.dump(label_encoders, os.path.join(PROCESSED_DIR, "label_encoders.pkl"))

    # Identify numeric columns (exclude target)
    target_col = "churn"
    numeric_cols = df.drop(columns=[target_col]).select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Scale numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save scaler and numeric column names
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))
    joblib.dump(numeric_cols, os.path.join(PROCESSED_DIR, "numeric_cols.pkl"))

    return df
