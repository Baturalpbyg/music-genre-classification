"""
Music Genre Classification - CLI Prediction Tool
------------------------------------------------
A command-line interface tool to predict music genres for local audio files.
Useful for quick testing without launching a web server.
"""

import librosa
import numpy as np
import pandas as pd
import joblib
import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- PATH CONFIGURATION ---
# Dynamically locate the 'dataset' folder relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

MODEL_PATH = os.path.join(DATASET_DIR, 'music_model.pkl')
SCALER_PATH = os.path.join(DATASET_DIR, 'music_scaler.pkl')

def extract_features(file_path):
    """
    Extracts features from the audio file.
    Must replicate the extraction logic from the training phase exactly.
    """
    try:
        y, sr = librosa.load(file_path, offset=40, duration=40)
        
        # Feature Extraction
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo

        data = {
            'tempo': [tempo_val],
            'spectral_centroid': [np.mean(cent)],
            'spectral_rolloff': [np.mean(rolloff)],
            'zero_crossing_rate': [np.mean(zcr)],
            'rms_energy': [np.mean(rms)]
        }
        
        for i, e in enumerate(np.mean(mfcc, axis=1)):
            data[f'mfcc_{i+1}'] = [e]
            
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"[ERROR] Could not process file: {e}")
        return None

def main():
    print("--------------------------------------------------")
    print("   AI MUSIC GENRE PREDICTOR (CLI)")
    print("--------------------------------------------------")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        print("Please run 'train_model.py' first.")
        sys.exit(1)

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("[INFO] Model loaded successfully.")
        print(f"[INFO] Supported Genres: {model.classes_}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # 2. Prediction Loop
    while True:
        print("\n" + "="*50)
        print("Enter the full path to an audio file (or 'q' to quit):")
        file_path = input("Path: ").strip().strip('"') # Remove quotes if user pastes as path
        
        if file_path.lower() == 'q':
            print("Exiting...")
            break
            
        if not os.path.exists(file_path):
            print("[ERROR] File does not exist. Please check the path.")
            continue
            
        print("\n[INFO] Analyzing audio...")
        input_data = extract_features(file_path)
        
        if input_data is not None:
            # Scale data
            input_data_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_data_scaled)[0]
            probabilities = model.predict_proba(input_data_scaled)[0]
            
            # Display Results
            print("-" * 30)
            print(f"PREDICTED GENRE: {prediction.upper()}")
            print("-" * 30)
            
            # Show Confidence
            sorted_indices = np.argsort(probabilities)[::-1]
            print("Confidence Levels:")
            for i in sorted_indices:
                print(f" - {model.classes_[i]}: {probabilities[i] * 100:.2f}%")

if __name__ == "__main__":
    main()