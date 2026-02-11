# 🎵 AI Music Genre Classifier

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

An end-to-end Machine Learning project that classifies audio files into musical genres using **Digital Signal Processing (DSP)** techniques. This system achieves **90% accuracy** by analyzing spectral and rhythmic features of audio signals.

## 🎯 Project Overview
This project was developed to explore the intersection of **Signal Processing** and **Machine Learning**. Instead of using "black box" Deep Learning models, I focused on **Feature Engineering**—mathematically extracting meaningful data from raw sound waves to train a robust Random Forest Classifier.

### Supported Genres:
* **Metal** 🤘
* **Classical** 🎻
* **Pop** 🎤
* **Blues** 🎷
* **Rap** 🎧

## 📸 Interface & Results

### 1. Real-Time Web Interface
The project includes a Streamlit-based web app for real-time analysis.
![Streamlit App Interface](images/app_interface.png)

### 2. Performance (90% Accuracy)
After implementing **Data Augmentation** and **Hyperparameter Tuning**, the model successfully distinguishes between complex genres (e.g., Metal vs. Classical).
![Confusion Matrix](images/confusion_matrix.png)

---

## ⚙️ The Engineering Behind It (How It Works)

The pipeline consists of three main stages:

### 1. Signal Processing & Decomposition
Using `Librosa`, raw audio is processed to separate **Harmonic** (melody) and **Percussive** (rhythm) components using **HPSS (Harmonic-Percussive Source Separation)**. This is crucial for distinguishing genres like Metal (percussive-heavy) from Classical (harmonic-heavy).

### 2. Advanced Feature Extraction
The system extracts a 58-dimensional feature vector for each track, including:
* **MFCCs (Mel-Frequency Cepstral Coefficients):** Represents the "timbre" or color of the sound.
* **Spectral Centroid & Bandwidth:** Measures the "brightness" and width of the sound.
* **Zero Crossing Rate (ZCR):** High ZCR correlates with noisy/distorted sounds (e.g., electric guitars).
* **Chroma Features:** Analyzes the harmonic/chord content.
* **Statistical Variance:** Captures the *dynamic range* of the track (not just the average).

### 3. Data Augmentation
To prevent overfitting and increase model robustness, the dataset was expanded using:
* **Noise Injection:** Simulating low-quality audio.
* **Time Stretching:** Altering tempo without changing pitch.
* **Pitch Shifting:** Changing key without altering tempo.

---

## 📂 Project Structure

```bash
music-genre-classification/
│
├── dataset/                # Dataset folder (Audio files + Models)
│   ├── music_dataset.csv   # Extracted features (The mathematical representation)
│   ├── music_model.pkl     # Trained Random Forest Model
│   └── music_scaler.pkl    # StandardScaler object
│
├── feature_extractor.py    # Extracts DSP features from raw audio
├── train_model.py          # Trains the model and generates the matrix
├── app.py                  # Streamlit Web Application
├── predict_cli.py          # Command Line Interface tool
├── requirements.txt        # Project dependencies
└── README.md               # Documentation