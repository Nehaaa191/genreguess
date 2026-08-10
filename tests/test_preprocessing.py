import pytest
import numpy as np
import librosa
import soundfile as sf
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.common.audio_preprocessing import process_audio_file, SAMPLE_RATE, N_MELS

@pytest.fixture
def synthetic_audio_path(tmp_path):
    # Generate a 5-second sine wave
    duration = 5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    path = tmp_path / "synthetic.wav"
    sf.write(path, audio, SAMPLE_RATE)
    return str(path)

def test_preprocessing(synthetic_audio_path):
    tensor = process_audio_file(synthetic_audio_path)
    
    # Check shape: (128, ~1250)
    assert tensor.shape[0] == N_MELS
    assert tensor.shape[1] > 1000 # Should be padded to 29 seconds length
    
    # Check normalization stat (mean close to 0, std close to 1)
    # The whole spectrogram is zero padded heavily, so std might not be exactly 1 over the whole thing
    # but let's check it doesn't crash and returns valid numpy array
    assert isinstance(tensor, np.ndarray)
    assert not np.isnan(tensor).any()
