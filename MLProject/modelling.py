import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def train_model():
    print("Starting Model Training (Basic Workflow with Autolog)...")
    
    # 1. Set MLflow tracking URI (Localhost Tracking)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Enable MLflow Autologging for Scikit-Learn
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True
    )
    
    # 2. Load preprocessed dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'heart_disease_preprocessed/heart_disease_preprocessed.csv')
    if not os.path.exists(dataset_path):
        # Fallback to current directory check
        dataset_path = 'heart_disease_preprocessed/heart_disease_preprocessed.csv'
        
    print(f"Loading preprocessed dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Split features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train Model within MLflow run
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        print("Training RandomForestClassifier...")
        # Define basic classifier without hyperparameter tuning
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        print(f"Model Training Complete.")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    train_model()
