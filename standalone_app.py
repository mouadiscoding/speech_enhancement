import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import io
import os
import requests
import torchaudio
from PIL import Image

FASTAPI_URL = "http://localhost:8080/denoise"

def record_audio(duration=5, sample_rate=16000):
    st.write("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording finished.")

    audio = (audio * 32767).astype(np.int16)
    audio = audio.squeeze()
    audio = audio / np.max(np.abs(audio))
    return audio, sample_rate

# Function to send audio to FastAPI and receive the denoised version
def denoise_audio(audio_data, sample_rate):
    files = {
        'file': ('noisy_audio.wav', audio_data, 'audio/wav')
    }
    params = {
        'sample_rate': sample_rate
    }
    print("Sending request and waiting for response...")
    response = requests.post(FASTAPI_URL, files=files, data=params)
    print("Response received!")
    print("Response status code:", response.status_code)
    return response


########################## Streamlit app ######################################

im = Image.open('./content/speaking-head.png')
st.set_page_config(page_title="Speech Denoising App", page_icon = im)

st.title("Audio Denoising App")
st.write("You can upload an audio file or record a new one to denoise it.")


option = st.selectbox("Choose an option", ("Record Audio", "Upload Audio"))

if option == "Record Audio":
    duration = st.slider("Recording duration (seconds)", 3, 10, 5)

    if st.button("Record"):
        audio, sample_rate = record_audio(duration=duration)
        st.audio(audio, sample_rate=sample_rate, format='audio/wav')

        wav_io = io.BytesIO()
        write(wav_io, sample_rate, audio)
        wav_io.seek(0)

        st.info("Denoising...")
        response = denoise_audio(wav_io, sample_rate)

        if response.status_code == 200:
            st.success("Denoising complete!")
            st.audio(response.content, format='audio/wav')
        else:
            st.error("Error during denoising")


elif option == "Upload Audio":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])
    
    if uploaded_file is not None:

        _, sample_rate = torchaudio.load(uploaded_file)
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Denoise"):
            st.info("Denoising...")
            response = denoise_audio(uploaded_file, sample_rate)

            if response.status_code == 200:
                st.success("Denoising complete!")
                st.audio(response.content, format='audio/wav')
            else:
                st.error("Error during denoising")

