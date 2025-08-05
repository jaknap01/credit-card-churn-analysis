# scripts/data_loader.py

import pandas as pd
import os

def load_data(file_path):
    """Load CSV data and return as DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\n Columns:")
    print(df.columns.tolist())
    print("\n Info:")
    print(df.info())
    print("\n Null values per column:")
    print(df.isnull().sum())
    
    return df

if __name__ == "__main__":
    # Adjust path if running from a different location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "data", "raw", "exl_credit_card_churn_data.csv")
    df = load_data(file_path)
