import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle

# Load the trained model, scaler, and label encoder
model = tf.keras.models.load_model("/app/model/audio_emotion_rnn.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Preprocess function
def preprocess_audio(audio_file):
    y, sr = librosa.load(audio_file, duration=2.5, offset=0.6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
    features = np.mean(mfcc.T, axis=0)
    features = scaler.transform([features])
    return features.reshape(1, features.shape[1], 1)

# Streamlit app UI
st.title("Audio Emotion Detection")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    features = preprocess_audio(uploaded_file)
    prediction = model.predict(features)
    emotion = label_encoder.inverse_transform([np.argmax(prediction)])
    st.write(f"Predicted Emotion: {emotion[0]}")
