import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('save_models/audio_classification.hdf5')

# Define classes for environmental sounds
classes = ['dog_bark', 'street_music', 'children_playing', 'car_horn', 'drilling','gun_shot']  # Update with your actual classes

def extract_features(file_path):
    # Load audio file using librosa
    audio, _ = librosa.load(file_path, res_type='kaiser_fast')

    # Extract MFCC features from the audio file
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    return mfccs_processed

def predict_class(audio_file):
    # Extract features from the audio file
    features = extract_features(audio_file)

    # Make prediction using the pre-trained model
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)

    # Get the predicted class
    predicted_class = classes[np.argmax(prediction)]

    return predicted_class

# Streamlit UI
st.title("Environmental Sound Classification App")

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio_file:
    st.audio(audio_file, format='audio/wav')

    # Make prediction when the user clicks the button
    if st.button("Classify"):
        prediction = predict_class(audio_file)
        st.success(f"The predicted class is: {prediction}")