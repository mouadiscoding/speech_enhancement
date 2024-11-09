import streamlit as st
from confluent_kafka import Consumer, KafkaError
import io
import time

KAFKA_BROKER = 'localhost:9092'
OUTPUT_TOPIC = 'clean'

# Configure the Kafka consumer
consumer = Consumer({
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'streamlit-audio-consumer',
    'auto.offset.reset': 'latest'  # Start from the latest message
})
consumer.subscribe([OUTPUT_TOPIC])

def receive_cleaned_audio():
    """
    Continuously polls Kafka for messages from the 'clean' topic.
    Returns the latest processed audio as bytes if available.
    """
    while True:
        msg = consumer.poll(1.0)  # Poll every second
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                st.error(f"Kafka error: {msg.error()}")
            continue
        return msg.value()

# Streamlit UI setup
st.title("Denoised Audio Playback")

# Placeholder to display audio when it arrives
audio_placeholder = st.empty()
status_placeholder = st.empty()

status_placeholder.info("Waiting for processed audio...")

# Continuously update the Streamlit app with new audio when it arrives
while True:
    cleaned_audio = receive_cleaned_audio()
    
    if cleaned_audio:
        audio_placeholder.audio(io.BytesIO(cleaned_audio), format="audio/wav")
        status_placeholder.success("Processed audio received.")
        
    time.sleep(1)  # Delay to prevent rapid polling and UI updates