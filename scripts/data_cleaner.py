import os
import pandas as pd
import numpy as np

# Get base directory (project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define dynamic paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'exl_credit_card_churn_data.csv')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_credit_card_churn_data.csv')

# Ensure directories exist
os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data', 'raw'), exist_ok=True)

# Load raw data
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Raw data file not found at {filepath}. Please ensure the file exists.")

# Data cleaning function
def clean_data(df):
    df['Gender'] = df['Gender'].str.capitalize()
    
    avg_age = np.floor(df[df['Age'] > 0]['Age'].mean())
    df['Age'] = np.where((df['Age'] <= 0) | (df['Age'] > 120), avg_age, df['Age'])
    
    binary_cols = ['HasCrCard', 'IsActiveMember']
    for col in binary_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[col] = df[col].map({'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Churn'] = df['Churn'].astype(str).str.lower().str.strip()
    df = df[~df['Churn'].isin(['maybe'])]
    df['Churn'] = df['Churn'].map({'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})
    df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
    
    df = df.dropna(subset=['HasCrCard', 'IsActiveMember', 'Churn'])
    
    numerical_cols = ['Tenure', 'Balance', 'EstimatedSalary']
    for col in numerical_cols:
        avg_value = np.floor(df[col].mean())
        df[col] = df[col].fillna(avg_value)
    
    mode_value = df['NumOfProducts'].mode()[0]
    df['NumOfProducts'] = df['NumOfProducts'].fillna(mode_value)
    
    gender_dist = df['Gender'].value_counts(normalize=True)
    df['Gender'] = df.apply(lambda x: np.random.choice(gender_dist.index, p=gender_dist.values)
                            if pd.isna(x['Gender']) else x['Gender'], axis=1)
    
    mean_age = df['Age'].mean()
    std_age = df['Age'].std()
    df['Age'] = df.apply(lambda x: np.random.normal(mean_age, std_age)
                         if pd.isna(x['Age']) else x['Age'], axis=1)
    df['Age'] = np.where(df['Age'] <= 0, avg_age, df['Age'])
    df['Age'] = df['Age'].round().astype(int)
    
    df[['HasCrCard', 'IsActiveMember', 'Churn']] = df[['HasCrCard', 'IsActiveMember', 'Churn']].astype(int)
    
    return df

# Main execution
if __name__ == "__main__":
    try:
        raw_df = load_data(RAW_DATA_PATH)
        print(f"Successfully loaded data from {RAW_DATA_PATH}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)
    
    cleaned_df = clean_data(raw_df)
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}")
    
    print("\nData Cleaning Report:")
    print(f"Original rows: {len(raw_df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print("\nNull values after cleaning:")
    print(cleaned_df.isnull().sum())
