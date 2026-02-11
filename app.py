"""
Music Genre Classification - Web Application
--------------------------------------------
A Streamlit-based web interface for the Music Genre Classification model.
Allows users to upload audio files and get real-time predictions
visualized with interactive charts.

Usage:
    Run via terminal: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import librosa
import joblib
import os
import plotly.express as px

# --- PATH CONFIGURATION ---
# Dynamically locate the 'dataset' folder relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

MODEL_PATH = os.path.join(DATASET_DIR, 'music_model.pkl')
SCALER_PATH = os.path.join(DATASET_DIR, 'music_scaler.pkl')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

# --- HEADER SECTION ---
st.title("🎵 AI Music Genre Classifier")
st.markdown("""
This application utilizes **Digital Signal Processing (DSP)** and **Machine Learning** techniques to analyze audio signals.
It predicts the music genre by processing rhythmic and spectral features through a trained **Random Forest Classifier**.
""")

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    """
    Loads the serialized model and scaler from the disk.
    Cached to prevent reloading on every interaction.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error(f"Model files not found in: {DATASET_DIR}")
        st.warning("Please run 'train_model.py' to generate the model files first.")
        return None, None

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None
def extract_features(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, offset=40, duration=30)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Extract Features
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    data = {
        'tempo': [tempo],
        'chroma_mean': [np.mean(chroma)], 'chroma_var': [np.var(chroma)],
        'rms_mean': [np.mean(rms)],       'rms_var': [np.var(rms)],
        'cent_mean': [np.mean(cent)],     'cent_var': [np.var(cent)],
        'rolloff_mean': [np.mean(rolloff)], 'rolloff_var': [np.var(rolloff)],
        'zcr_mean': [np.mean(zcr)],       'zcr_var': [np.var(zcr)],
        'bw_mean': [np.mean(bw)],         'bw_var': [np.var(bw)],
    }
    
    for i in range(1, 21):
        data[f'mfcc_{i}_mean'] = [np.mean(mfcc[i-1])]
        data[f'mfcc_{i}_var'] = [np.var(mfcc[i-1])]
        
    return pd.DataFrame(data)

# --- MAIN APPLICATION LOGIC ---
model, scaler = load_model()

if model is not None:
    # File Uploader Widget
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Display Audio Player
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner('Analyzing audio signal...'):
            try:
                # 1. Feature Extraction
                input_data = extract_features(uploaded_file)
                
                # 2. Preprocessing (Scaling)
                input_data_scaled = scaler.transform(input_data)
                
                # 3. Prediction
                prediction = model.predict(input_data_scaled)[0]
                probabilities = model.predict_proba(input_data_scaled)[0]
                
                # 4. Display Results
                st.success(f"Prediction: **{prediction.upper()}**")
                
                # Interactive Bar Chart
                df_prob = pd.DataFrame({
                    'Genre': model.classes_,
                    'Confidence': probabilities
                })
                
                fig = px.bar(
                    df_prob, 
                    x='Genre', 
                    y='Confidence', 
                    color='Genre', 
                    title="Model Confidence Scores",
                    text_auto='.2%',
                    template="plotly_white"
                )
                st.plotly_chart(fig)
                
                # Engineering Data Expander
                with st.expander("View DSP Feature Data"):
                    st.write("Extracted numerical features:")
                    st.dataframe(input_data)
            
            except Exception as e:
                st.error(f"Error processing file: {e}")