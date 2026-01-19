"""
Script to retrain the models with the current scikit-learn version.
Run this script to regenerate the model files compatible with your current environment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

print("Loading data...")
data = pd.read_csv("feature_engineered_greenpulse.csv")

print("Preparing features...")
target = "GDI"
X = data.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Regression Model...")
rf_regressor = RandomForestRegressor(random_state=42, n_estimators=100)
rf_regressor.fit(X_train, y_train)
y_pred_rf = rf_regressor.predict(X_test)

print(f"Regression R² Score: {r2_score(y_test, y_pred_rf):.3f}")
print(f"Regression MAE: {mean_absolute_error(y_test, y_pred_rf):.3f}")

print("\nTraining Classification Model...")
def categorize_gdi(value):
    if value <= -5:
        return "Excellent (Net Gain)"
    elif -5 < value <= 0:
        return "Acceptable"
    elif 0 < value <= 10:
        return "Concerning"
    else:
        return "High-Risk"

data["GDI_Category"] = data["GDI"].apply(categorize_gdi)
y_class = data["GDI_Category"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train_c, y_train_c)
y_pred_rf_clf = rf_classifier.predict(X_test_c)

print(f"Classification Accuracy: {accuracy_score(y_test_c, y_pred_rf_clf):.3f}")
print(f"Classification F1: {f1_score(y_test_c, y_pred_rf_clf, average='weighted'):.3f}")

print("\nSaving models...")
joblib.dump(rf_regressor, "random_forest_regressor.pkl")
joblib.dump(rf_classifier, "random_forest_classifier.pkl")

print("✓ Models saved successfully!")
print("  - random_forest_regressor.pkl")
print("  - random_forest_classifier.pkl")
print("\nYou can now restart your Flask app and the models should load correctly.")

