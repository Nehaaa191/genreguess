import io
import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
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
    
    try:
        with io.BytesIO(file_bytes) as f:
            info = sf.info(f)
            sr_loaded = info.samplerate
            
            samples_per_track = sr * config["duration"]
            samples_per_track_orig = int(sr_loaded * config["duration"])
            total_samples = info.frames
            
            if total_samples > samples_per_track_orig:
                start_frame = (total_samples - samples_per_track_orig) // 2
                frames_to_read = samples_per_track_orig
            else:
                start_frame = 0
                frames_to_read = total_samples
                
            f.seek(0)
            waveform, _ = sf.read(f, start=start_frame, frames=frames_to_read, dtype='float32', always_2d=True)
            
    except Exception as e:
        raise ValueError(f"Failed to read audio file: {e}")
        
    # Convert to mono
    if waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1)
    else:
        waveform = waveform.squeeze()
        
    # Resample if needed
    if sr_loaded != sr:
        waveform = librosa.resample(waveform, orig_sr=sr_loaded, target_sr=sr)
        
    if len(waveform) < sr * 0.1:  # less than 0.1 seconds
        raise ValueError("Audio is too short.")
        
    # 1. Pad if necessary (cropping is already done by soundfile)
    if len(waveform) < samples_per_track:
        pad_width = int(samples_per_track - len(waveform))
        waveform = np.pad(waveform, (0, pad_width), mode='constant')
    elif len(waveform) > samples_per_track:
        waveform = waveform[:int(samples_per_track)]
        
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

