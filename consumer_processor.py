from confluent_kafka import Consumer, Producer, KafkaError
import requests
import io

KAFKA_BROKER = 'localhost:9092'
INPUT_TOPIC = 'noisy'
OUTPUT_TOPIC = 'clean'
FASTAPI_URL = "http://localhost:8080/denoise"

consumer = Consumer({
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'audio-processing-group',
    'auto.offset.reset': 'earliest'
})
producer = Producer({'bootstrap.servers': KAFKA_BROKER})
consumer.subscribe([INPUT_TOPIC])

# Function to send audio to FastAPI and receive the denoised version
def denoise_audio(audio_data):
    # Wrap the byte data in a BytesIO object to simulate a file upload
    audio_buffer = io.BytesIO(audio_data)
    
    # Prepare the files dictionary to send as form data
    files = {
        'file': ('noisy_audio.wav', audio_buffer, 'audio/wav')
    }
    params = {
        'sample_rate': 16000  # Adjust if necessary
    }
    
    print("Sending request and waiting for response...")
    response = requests.post(FASTAPI_URL, files=files, data=params)
    print("Response status code:", response.status_code)
    
    if response.status_code == 200:
        return response.content  # Return the cleaned audio
    else:
        print("Error during denoising:", response.text)
        return None

def consume_and_process():
    while True:
        # Poll for messages from Kafka
        msg = consumer.poll(1.0)
        
        if msg is None:
            continue
        
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print("Error:", msg.error())
            continue
        
        # Get the noisy audio data from Kafka message
        audio_data = msg.value()
        
        # Denoise the audio
        cleaned_audio = denoise_audio(audio_data)
        
        if cleaned_audio:
            # Send the cleaned audio back to Kafka's output topic
            producer.produce(OUTPUT_TOPIC, value=cleaned_audio)
            producer.flush()
            print("Processed audio sent to output topic.")

if __name__ == "__main__":
    consume_and_process()