# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load trained objects
# -----------------------------
MODEL_PATH = "models/best_model.pkl"
FEATURES_PATH = "models/model_features.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/label_encoders.pkl"
NUMERIC_PATH = "models/numeric_cols.pkl"
DATA_PATH = "data/raw/bank_churn.csv"

model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
numeric_cols = joblib.load(NUMERIC_PATH)

# Load raw data for dynamic inputs
raw_df = pd.read_csv(DATA_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details below:")

# Define which columns are actually binary Yes/No
binary_cols = ["HasCrCard", "IsActiveMember"]  # adjust based on your dataset

input_dict = {}

for feature in model_features:
    if feature in binary_cols:  # True binary
        input_dict[feature] = st.selectbox(f"{feature}?", ["Yes", "No"])
    elif raw_df[feature].dtype == object:  # other categorical
        options = raw_df[feature].dropna().unique().tolist()
        input_dict[feature] = st.selectbox(f"{feature}", options)
    else:  # numeric
        min_val = int(raw_df[feature].min())
        max_val = int(raw_df[feature].max())
        default_val = int(raw_df[feature].mean())
        input_dict[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=default_val)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):
    input_data = pd.DataFrame([input_dict])

    # Encode binary categorical features
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].map({"Yes": 1, "No": 0})

    # Encode other categorical features using LabelEncoder if needed
    for col in raw_df.select_dtypes(include='object').columns:
        if col in input_data.columns and col not in label_encoders:
            le = LabelEncoder()
            le.fit(raw_df[col])
            input_data[col] = le.transform(input_data[col])

    # Ensure all model features exist and correct order
    input_data = input_data.reindex(columns=model_features, fill_value=0)

    # Scale numeric features
    for col in numeric_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"❌ This customer is likely to **CHURN** (Probability: {probability:.2f})")
    else:
        st.success(f"✅ This customer will **STAY** (Probability of Churn: {probability:.2f})")
