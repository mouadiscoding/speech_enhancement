import streamlit as st
from confluent_kafka import Producer


def produce_audio(audio_bytes):
    producer.produce(INPUT_TOPIC, audio_bytes)
    producer.flush()
    st.success("Audio sent to Kafka for processing")


# Kafka Configuration
KAFKA_BROKER = "localhost:9092"
INPUT_TOPIC = "noisy"

# Initialize Kafka Producer
producer_conf = {'bootstrap.servers': KAFKA_BROKER}
producer = Producer(producer_conf)


# Streamlit App
st.title("Real-Time Audio Cleaning App with Kafka")
st.write("Upload an audio file to be processed in real-time by the API via Kafka.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    audio_data = uploaded_file.read()
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Send for Denoising"):
        produce_audio(audio_data)