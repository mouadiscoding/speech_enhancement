import subprocess
import time
import signal
import atexit


## Create Topics (Do only once):
# ./bin/windows/kafka-topics.bat --create --topic noisy --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
# ./bin/windows/kafka-topics.bat --create --topic clean --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Build Docker Image if necessary
# docker build -t speech_denoise_image .

# Run Docker image
# docker run --gpus all --name speech_denoise_container -p 8080:8080 speech_denoise_image

# Start Zookeeper and Kafka Server
# Start Zookeeper : ./bin/windows/zookeeper-server-start.bat ./config/zookeeper.properties
# Start Kafka Server : ./bin/windows/kafka-server-start.bat config/server.properties

# List to keep track of subprocesses
processes = []

def start_process(command):
    """Start a subprocess and add it to the processes list."""
    process = subprocess.Popen(command, shell=True)
    processes.append(process)
    return process

def cleanup_processes():
    """Terminate all subprocesses on exit."""
    for process in processes:
        process.terminate()
        process.wait()

# Register the cleanup handler
atexit.register(cleanup_processes)

if __name__ == "__main__":
    try:
        print("Starting Streamlit producer app...")
        start_process("streamlit run producer.py")

        time.sleep(2)  # Short delay to stagger starts

        print("Starting Kafka consumer and processor...")
        start_process("python consumer_processor.py")

        time.sleep(2)  # Short delay to stagger starts

        print("Starting Streamlit consumer app...")
        start_process("streamlit run consumer.py")

        # Wait for all subprocesses to complete
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cleanup_processes()