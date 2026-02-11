import librosa
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_CSV = os.path.join(DATASET_PATH, "music_dataset.csv")

def extract_features(file_path):
    """
    Advanced Feature Extraction with Smart Loading
    Prevents errors on short/augmented files.
    """
    try:
       
        total_duration = librosa.get_duration(path=file_path)
        
       
        if total_duration < 3:
            print(f"⚠️ Skipped (Too short): {os.path.basename(file_path)}")
            return None

        
        start_offset = 40 if total_duration > 50 else 0
        
      
        y, sr = librosa.load(file_path, offset=start_offset, duration=30)
       
        if len(y) < sr * 5: 
            return None

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        data = {
            'tempo': tempo,
            'chroma_mean': np.mean(chroma), 'chroma_var': np.var(chroma),
            'rms_mean': np.mean(rms),       'rms_var': np.var(rms),
            'cent_mean': np.mean(cent),     'cent_var': np.var(cent),
            'rolloff_mean': np.mean(rolloff), 'rolloff_var': np.var(rolloff),
            'zcr_mean': np.mean(zcr),       'zcr_var': np.var(zcr),
            'bw_mean': np.mean(bw),         'bw_var': np.var(bw),
        }
        
        for i in range(1, 21):
            data[f'mfcc_{i}_mean'] = np.mean(mfcc[i-1])
            data[f'mfcc_{i}_var'] = np.var(mfcc[i-1])
            
        return data
        
    except Exception as e:
        # Hata olsa bile kod durmasın, sadece ekrana yazsın ve devam etsin
        print(f"❌ ERROR processing {os.path.basename(file_path)}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🚀 Starting ROBUST data analysis... ☕")
    
    all_data = []

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset folder '{DATASET_PATH}' not found!")
        exit()

    genres = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    for genre in genres:
        genre_folder = os.path.join(DATASET_PATH, genre)
        print(f"\n--- Processing: {genre.upper()} ---")
        
        songs = [f for f in os.listdir(genre_folder) if f.endswith(('.wav', '.mp3'))]
        
        
        count = 0
        for song in songs:
            file_full_path = os.path.join(genre_folder, song)
            features = extract_features(file_full_path)
            
            if features:
                features['genre'] = genre 
                features['filename'] = song
                all_data.append(features)
                count += 1
                
                if count % 10 == 0: print(".", end="", flush=True)
        
        print(f"\n✅ {count} tracks processed for {genre}.")

    if len(all_data) > 0:
        df = pd.DataFrame(all_data)
        
        cols = ['filename', 'genre'] + [c for c in df.columns if c not in ['filename', 'genre']]
        df = df[cols]
        
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 DATASET SAVED: {OUTPUT_CSV}")
        print(f"Total tracks in dataset: {len(df)}")
    else:
        print("\n❌ No data extracted.")