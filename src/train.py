"""
Train a model to predict if a resume matches a job description.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from preprocessing import load_data, extract_features, prepare_dataset

def train_model(X, y, model_type="random_forest"):
    """Train a model with hyperparameter tuning."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define model and parameters
    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    else:  # logistic regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear', 'saga']
        }
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=2, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    return best_model, X_test, y_test, y_pred

def save_model(model, model_path="data/model.joblib"):
    """Save the trained model."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def save_results(X_test, y_test, y_pred, results_path="data/results.json"):
    """Save test results for later analysis."""
    results = {
        "true_labels": y_test.tolist(),
        "predictions": y_pred.tolist()
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Results saved to {results_path}")

def main():
    """Main training pipeline."""
    print("Loading data...")
    df = load_data()
    if df is None:
        return
    
    print("Extracting features...")
    features, _ = extract_features(df)
    
    print("Preparing dataset...")
    X, y = prepare_dataset(df, features)
    
    if X is None or y is None:
        return
    
    print("Training model...")
    model, X_test, y_test, y_pred = train_model(X, y, model_type="random_forest")
    
    print("Saving model and results...")
    save_model(model)
    save_results(X_test, y_test, y_pred)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
