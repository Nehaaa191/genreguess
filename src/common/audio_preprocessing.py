import librosa
import numpy as np

# Blueprint parameters
SAMPLE_RATE = 22050
DURATION = 29  # 29 seconds window
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
POWER = 2.0

def load_and_crop_audio(filepath: str) -> np.ndarray:
    """
    Loads an audio file and extracts a fixed 29-second window from the center.
    If the file is shorter, it is zero-padded.
    """
    # Load audio, librosa returns mono and resamples if needed
    waveform, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    
    # Crop or pad to EXACTLY 29 seconds
    # If longer: take center crop
    if len(waveform) > SAMPLES_PER_TRACK:
        start = (len(waveform) - SAMPLES_PER_TRACK) // 2
        waveform = waveform[start : start + SAMPLES_PER_TRACK]
    # If shorter: zero-pad
    elif len(waveform) < SAMPLES_PER_TRACK:
        pad_width = SAMPLES_PER_TRACK - len(waveform)
        # Pad at the end
        waveform = np.pad(waveform, (0, pad_width), mode='constant')
        
    return waveform

def normalize_amplitude(waveform: np.ndarray) -> np.ndarray:
    """Normalizes the waveform amplitude to the range [-1, 1]."""
    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp
    return waveform

def compute_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """
    Computes a log-mel spectrogram from a waveform.
    Returns array of shape (N_MELS, T)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, 
        sr=SAMPLE_RATE, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS,
        power=POWER
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def standardize_spectrogram(log_mel_spec: np.ndarray) -> np.ndarray:
    """
    Per-spectrogram normalization: zero mean and unit variance.
    Alternatively, this could use global stats. For this implementation,
    we use per-spectrogram standardization which helps CNN stability 
    and simplifies inference since we don't depend on global training stats.
    """
    mean = np.mean(log_mel_spec)
    std = np.std(log_mel_spec)
    if std > 0:
        return (log_mel_spec - mean) / std
    return log_mel_spec - mean

def process_audio_file(filepath: str) -> np.ndarray:
    """
    End-to-end preprocessing pipeline for a single audio file.
    Output shape: (128, T) where T is approx 1250 (depends on hop_length).
    """
    waveform = load_and_crop_audio(filepath)
    waveform = normalize_amplitude(waveform)
    log_mel_spec = compute_mel_spectrogram(waveform)
    norm_spec = standardize_spectrogram(log_mel_spec)
    return norm_spec
