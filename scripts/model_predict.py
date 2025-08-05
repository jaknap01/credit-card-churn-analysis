import os
import pandas as pd
import numpy as np
import joblib

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
LOGREG_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'logistic_model.pkl')
RF_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'random_forest_model.pkl')
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'features_credit_card_churn.csv')

# Columns that were scaled during training
NUMERIC_COLS = ['Age', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts']

# Load all required models and scaler
def load_objects():
    scaler = joblib.load(SCALER_PATH)
    logreg_model = joblib.load(LOGREG_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    return scaler, logreg_model, rf_model

# Sample input (3 cases)
def get_sample_input():
    return pd.DataFrame([
        {
            'CustomerID': 'CUST9999',
            'Gender': 'Male',
            'Age': 45,
            'Tenure': 5,
            'Balance': 50000,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 75000
        },
        {
            'CustomerID': 'CUST1000',
            'Gender': 'Female',
            'Age': 30,
            'Tenure': 3,
            'Balance': 10000,
            'NumOfProducts': 1,
            'HasCrCard': 0,
            'IsActiveMember': 0,
            'EstimatedSalary': 50000
        },
        {
            'CustomerID': 'CUST1001',
            'Gender': 'Male',
            'Age': 60,
            'Tenure': 9,
            'Balance': 120000,
            'NumOfProducts': 3,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 95000
        }
    ])

# Preprocess input to match training
def preprocess_input(sample_df, reference_columns, scaler):
    df = sample_df.copy()

    # Fix gender
    df['Gender'] = df['Gender'].str.strip().str.capitalize()
    df = pd.get_dummies(df, columns=['Gender'])

    # Add any missing dummy columns
    for col in reference_columns:
        if col not in df.columns and col not in ['CustomerID', 'Churn']:
            df[col] = 0

    # Keep only feature columns in same order
    feature_cols = [col for col in reference_columns if col not in ['CustomerID', 'Churn']]
    df = df[feature_cols]

    # Scale only numeric columns
    df_scaled = df.copy()
    df_scaled[NUMERIC_COLS] = scaler.transform(df_scaled[NUMERIC_COLS])

    return df_scaled

if __name__ == "__main__":
    try:
        print("Loading models and scaler...")
        scaler, logreg_model, rf_model = load_objects()

        print("Loading reference training columns...")
        reference_df = pd.read_csv(TRAIN_DATA_PATH)
        reference_columns = reference_df.columns

        print("Generating test samples...")
        sample = get_sample_input()

        print("Preprocessing input data...")
        sample_processed = preprocess_input(sample, reference_columns, scaler)

        print("\n=== Prediction Output ===")
        for idx, row in sample.iterrows():
            cid = row['CustomerID']
            logreg_pred = logreg_model.predict(sample_processed.iloc[[idx]])[0]
            rf_pred = rf_model.predict(sample_processed.iloc[[idx]])[0]

            print(f"\nCustomerID: {cid}")
            print(f"  Logistic Regression: {'Churn' if logreg_pred == 1 else 'No Churn'}")
            print(f"  Random Forest:       {'Churn' if rf_pred == 1 else 'No Churn'}")

    except Exception as e:
        print(f"Prediction failed: {str(e)}")
