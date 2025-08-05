import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'features_credit_card_churn.csv')

MODEL_DIR = os.path.join(BASE_DIR, 'model')
LOGREG_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_model.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.txt')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    return pd.read_csv(filepath)

def evaluate_model(model, X_test, y_test, model_name: str) -> str:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    report = (
        f"=== {model_name} ===\n"
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        f"Confusion Matrix:\n{cm}\n"
        f"Classification Report:\n{cr}\n\n"
    )
    return report

if __name__ == "__main__":
    try:
        print("Loading data...")
        df = load_data(DATA_PATH)

        # Drop non-feature columns
        if "CustomerID" in df.columns:
            X = df.drop(["CustomerID", "Churn"], axis=1)
        else:
            X = df.drop("Churn", axis=1)

        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train logistic regression
        print("Training Logistic Regression...")
        logreg_model = LogisticRegression(max_iter=1000)
        logreg_model.fit(X_train, y_train)
        joblib.dump(logreg_model, LOGREG_MODEL_PATH)

        # Train random forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, RF_MODEL_PATH)

        # Evaluate both models
        print("Evaluating models...")
        logreg_report = evaluate_model(logreg_model, X_test, y_test, "Logistic Regression")
        rf_report = evaluate_model(rf_model, X_test, y_test, "Random Forest")

        # Save combined report
        with open(METRICS_PATH, "w") as f:
            f.write(logreg_report)
            f.write(rf_report)

        print("Models and metrics saved successfully.\n")
        print("=== Logistic Regression vs Random Forest ===")
        print(logreg_report)
        print(rf_report)

    except Exception as e:
        print(f"Error during training: {str(e)}")
