# predict.py
import joblib
import pandas as pd
import os

MODEL_DIR = "models/"

def load_artifacts():
    """Load saved model, encoders, and scaler"""
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return model, label_encoders, scaler

def preprocess_input(new_data, label_encoders, scaler):
    """Encode and scale new data for prediction"""
    for col, le in label_encoders.items():
        new_data[col] = le.transform(new_data[col])
    X_scaled = scaler.transform(new_data)
    return X_scaled

def predict(new_data):
    """Make predictions on new input"""
    model, label_encoders, scaler = load_artifacts()
    processed = preprocess_input(new_data, label_encoders, scaler)
    preds = model.predict(processed)
    return preds

if __name__ == "__main__":
    # Example input (same format as dataset, except 'churn')
    sample = pd.DataFrame([{
        "customer_id": 15678900,
        "credit_score": 650,
        "country": "France",
        "gender": "Male",
        "age": 35,
        "tenure": 5,
        "balance": 60000,
        "products_number": 2,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 50000
    }])

    prediction = predict(sample)
    print("✅ Prediction:", prediction)
