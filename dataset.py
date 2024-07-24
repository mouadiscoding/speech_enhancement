# Adapted from https://github.com/NVIDIA/CleanUNet under the MIT License.

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

import soundfile as sf

from torchvision import datasets, models, transforms
import torchaudio

class CleanNoisyDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, sample_rate=16000, max_length=48000): # max length = 10 seconds at 16kHz
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        self.clean_files = sorted(os.listdir(clean_dir))
        self.noisy_files = sorted(os.listdir(noisy_dir))
        
        assert len(self.clean_files) == len(self.noisy_files), "Mismatch in number of clean and noisy files"

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        
        # Load audio files using soundfile
        clean_waveform, sr_clean = sf.read(clean_path)
        noisy_waveform, sr_noisy = sf.read(noisy_path)
        
        # Convert to torch tensors
        clean_waveform = torch.FloatTensor(clean_waveform)
        noisy_waveform = torch.FloatTensor(noisy_waveform)
        
        clean_waveform, noisy_waveform = clean_waveform.squeeze(0), noisy_waveform.squeeze(0)

        assert len(clean_waveform) == len(noisy_waveform)

        # Resample if necessary
        if sr_clean != self.sample_rate:
            clean_waveform = torchaudio.transforms.Resample(sr_clean, self.sample_rate)(clean_waveform)
        if sr_noisy != self.sample_rate:
            noisy_waveform = torchaudio.transforms.Resample(sr_noisy, self.sample_rate)(noisy_waveform)
        
        # Pad or truncate to max_length
        if clean_waveform.size(0) < self.max_length:
            clean_waveform = torch.nn.functional.pad(clean_waveform, (0, self.max_length - clean_waveform.size(0)))
        else:
            clean_waveform = clean_waveform[:self.max_length]
        
        if noisy_waveform.size(0) < self.max_length:
            noisy_waveform = torch.nn.functional.pad(noisy_waveform, (0, self.max_length - noisy_waveform.size(0)))
        else:
            noisy_waveform = noisy_waveform[:self.max_length]

        # Add channel dimension
        clean_waveform = clean_waveform.unsqueeze(0)
        noisy_waveform = noisy_waveform.unsqueeze(0)
        
        return clean_waveform, noisy_waveform

def load_CleanNoisyPairDataset(clean_dir, noisy_dir, sample_rate=16000, batch_size=32, shuffle=True, num_workers=0):
    """
    Load and return the CleanNoisyDataset along with its DataLoader.

    Args:
    - clean_dir (str): Path to the directory containing clean audio files.
    - noisy_dir (str): Path to the directory containing noisy audio files.
    - sample_rate (int): Target sample rate for the audio. Default is 16000.
    - batch_size (int): Batch size for the DataLoader. Default is 32.
    - shuffle (bool): Whether to shuffle the data. Default is True.
    - num_workers (int): Number of worker processes for data loading. Default is 4.

    Returns:
    - dataset (CleanNoisyDataset): The created dataset.
    - dataloader (DataLoader): DataLoader for the dataset.
    """
    dataset = CleanNoisyDataset(clean_dir, noisy_dir, sample_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataset, dataloader