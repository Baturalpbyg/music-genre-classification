"""
Data Augmentation Script
------------------------
Generates synthetic audio data to increase dataset size.
Techniques: Noise Injection, Time Stretching, Pitch Shifting.
"""

import librosa
import numpy as np
import soundfile as sf
import os
import warnings
from tqdm import tqdm  # İlerleme çubuğu için

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

def add_noise(y):
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    return y + noise_amp * np.random.normal(size=y.shape)

def change_speed(y):
    rate = np.random.uniform(low=0.9, high=1.1)
    return librosa.effects.time_stretch(y=y, rate=rate)

def change_pitch(y, sr):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)

def process_augmentation():
    print(f"📂 Scanning dataset at: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        print("ERROR: Dataset folder not found.")
        return

    # Klasörleri bul
    genres = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    total_new_files = 0
    
    for genre in genres:
        genre_path = os.path.join(DATASET_DIR, genre)
        songs = [f for f in os.listdir(genre_path) if f.endswith(('.wav', '.mp3')) and "aug_" not in f]
        
        print(f"\n⚡ Augmenting {genre.upper()} ({len(songs)} original tracks)...")
        
        for song in tqdm(songs):
            file_path = os.path.join(genre_path, song)
            
            try:
                # 30 sn oku
                y, sr = librosa.load(file_path, duration=30)
                
                # 1. Noise
                y_noise = add_noise(y)
                sf.write(os.path.join(genre_path, f"aug_noise_{song}.wav"), y_noise, sr)
                
                # 2. Speed
                y_speed = change_speed(y)
                sf.write(os.path.join(genre_path, f"aug_speed_{song}.wav"), y_speed, sr)
                
                # 3. Pitch
                y_pitch = change_pitch(y, sr)
                sf.write(os.path.join(genre_path, f"aug_pitch_{song}.wav"), y_pitch, sr)
                
                total_new_files += 3
                
            except Exception as e:
                print(f"Error on {song}: {e}")

    print(f"\n🎉 SUCCESS! Generated {total_new_files} new synthetic samples.")
    print("Now re-run 'feature_extractor.py' to include these files!")

if __name__ == "__main__":
    process_augmentation()