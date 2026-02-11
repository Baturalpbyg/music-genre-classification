"""
Music Genre Classification - Model Training Script
--------------------------------------------------
This script loads the extracted audio features from the CSV file,
cleans the data, trains an OPTIMIZED Random Forest Classifier,
and saves the model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import sys

# Scikit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONFIGURATION & PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

CSV_FILE_PATH = os.path.join(DATASET_DIR, "music_dataset.csv")
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'music_model.pkl')
SCALER_SAVE_PATH = os.path.join(DATASET_DIR, 'music_scaler.pkl')

def main():
    # --- 1. LOAD DATASET ---
    print(f"[INFO] Loading dataset from: {CSV_FILE_PATH}")
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"[ERROR] CSV file not found at {CSV_FILE_PATH}")
        sys.exit(1)

    try:
        data = pd.read_csv(CSV_FILE_PATH)
        
        # --- 🚑 BUG FIX: CLEANING TEMPO COLUMN ---
        if data['tempo'].dtype == 'O': 
            print("[INFO] Detected string formatting in 'tempo' column. Cleaning...")
            data['tempo'] = data['tempo'].astype(str).str.replace('[', '', regex=False)
            data['tempo'] = data['tempo'].str.replace(']', '', regex=False)
            data['tempo'] = data['tempo'].astype(float)
            print("[INFO] 'tempo' column fixed successfully.")
            
        print(f"[INFO] Successfully loaded {len(data)} samples.")
        
    except Exception as e:
        print(f"[ERROR] Failed to process CSV file: {e}")
        sys.exit(1)

    # --- 2. DATA PREPROCESSING ---
    # Separate features (X) and target labels (y)
    X = data.drop(['filename', 'genre'], axis=1)
    y = data['genre']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[INFO] Training set size: {len(X_train)} samples")
    print(f"[INFO] Test set size: {len(X_test)} samples")

    # Feature Scaling
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except ValueError as e:
        print(f"\n[CRITICAL ERROR] Data scaling failed: {e}")
        sys.exit(1)

    # --- 3. MODEL TRAINING (OPTIMIZED) ---
    print("[INFO] Initializing Optimized Random Forest Classifier...")
    
  
    model = RandomForestClassifier(
        n_estimators=200,    
        max_depth=15,        
        random_state=42,
        n_jobs=-1            
    )
    
    print("[INFO] Training model...")
    model.fit(X_train_scaled, y_train)
    print("[INFO] Training complete.")

    # --- 4. EVALUATION ---
    print("[INFO] Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Model Accuracy: {accuracy * 100:.2f}%")

    # Detailed Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("[INFO] Generating confusion matrix...")
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # --- 5. MODEL SERIALIZATION ---
    print(f"[INFO] Saving model artifacts to {DATASET_DIR}...")
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print("[SUCCESS] Model and scaler saved successfully.")

if __name__ == "__main__":
    main()