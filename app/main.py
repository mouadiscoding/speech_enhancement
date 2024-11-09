from fastapi import FastAPI, File, UploadFile
import sys
import torchaudio
from io import BytesIO
from fastapi.responses import FileResponse
from denoise import cleanunet_denoise_single_sample

app = FastAPI()


@app.get("/")
def root():
    return{"message": "Speech Denoiser API"}

@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(...), sample_rate: int = 16000):
    """
    Denoises the uploaded noisy audio file using the pretrained model.
    """
    # Load the uploaded file into memory
    file_bytes = await file.read()
    
    # Load the noisy audio into a tensor
    noisy_audio, sr = torchaudio.load(BytesIO(file_bytes))
    
    # Ensure the sample rate matches the model's sample rate
    if sr != sample_rate:
        noisy_audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(noisy_audio)
    
    model_path = "/code/app/best_model.pkl"

    # Denoise the audio
    denoised_audio = cleanunet_denoise_single_sample(noisy_audio, model_path, sample_rate)
    denoised_audio = denoised_audio.squeeze(0)
    # Save the denoised audio to a temporary file
    output_file_path = "denoised_audio.wav"
    torchaudio.save(output_file_path, denoised_audio, sample_rate)
    
    # Return the denoised audio file as a response
    return FileResponse(output_file_path, media_type="audio/wav", filename="denoised_audio.wav")