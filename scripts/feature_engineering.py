import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_credit_card_churn_data.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'features_credit_card_churn.csv')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

# Ensure necessary directories exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

def load_data(filepath: str) -> pd.DataFrame:
    """Load cleaned data from CSV"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at {filepath}")
    return pd.read_csv(filepath)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations"""
    df = df.copy()

    # Step 1: Remove any pre-existing dummy columns from previous runs
    dummy_cols = [col for col in df.columns if col.lower().startswith('gender_')]
    df.drop(columns=dummy_cols, errors='ignore', inplace=True)

    # Step 2: One-Hot Encoding for Gender (only Gender_Male will be kept)
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # Step 3: Normalize numerical columns
    numeric_cols = ['Age', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts']
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Step 4: Save the scaler for future use
    joblib.dump(scaler, SCALER_PATH)

    return df

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save processed data to CSV"""
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    try:
        print("Starting feature engineering...")

        # Load data
        print(f"Loading data from {INPUT_PATH}")
        data = load_data(INPUT_PATH)

        # Engineer features
        print("Engineering features...")
        featured_data = engineer_features(data)

        # Save final features
        print(f"Saving featured data to {OUTPUT_PATH}")
        save_data(featured_data, OUTPUT_PATH)

        print("Feature engineering completed successfully!")
        print("\nPreview of featured data:")
        print(featured_data.head())

    except Exception as e:
        print(f"Error during feature engineering: {str(e)}")
