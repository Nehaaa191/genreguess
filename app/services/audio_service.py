import io
import os
import sys
import torch
import numpy as np
import librosa
from app.config import PREPROCESSING_CONFIG_PATH
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.audio_preprocessing import (
    load_and_crop_audio, 
    normalize_amplitude, 
    compute_mel_spectrogram, 
    standardize_spectrogram
)

def process_audio_bytes(file_bytes: bytes) -> torch.Tensor:
    # Read the JSON config to assert/match parameters
    with open(PREPROCESSING_CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    sr = config["sample_rate"]
    
    # librosa.load can read from file-like objects using soundfile backend
    # but to be safe with any format librosa supports, we can write to temp
    # However, BytesIO is often fine if it's a standard WAV
    try:
        waveform, sr_loaded = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to read audio file: {e}")
        
    if len(waveform) < sr * 0.1:  # less than 0.1 seconds
        raise ValueError("Audio is too short.")
        
    # We must use exactly the same logic as training
    samples_per_track = sr * config["duration"]
    
    # 1. Crop/Pad
    if len(waveform) > samples_per_track:
        start = (len(waveform) - samples_per_track) // 2
        waveform = waveform[start : start + samples_per_track]
    elif len(waveform) < samples_per_track:
        pad_width = samples_per_track - len(waveform)
        waveform = np.pad(waveform, (0, pad_width), mode='constant')
        
    # 2. Normalize
    waveform = normalize_amplitude(waveform)
    
    # 3. Mel Spec (assuming the config matches the hardcoded ones in src/common)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        power=config["power"]
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 4. Standardize
    norm_spec = standardize_spectrogram(log_mel_spec)
    
    # Convert to Tensor (1, 1, 128, T)
    tensor = torch.tensor(norm_spec, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor
