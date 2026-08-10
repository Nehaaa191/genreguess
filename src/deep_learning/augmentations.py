import numpy as np
import random

def spec_augment(log_mel_spec: np.ndarray, time_mask_param=20, freq_mask_param=15, num_time_masks=2, num_freq_masks=2) -> np.ndarray:
    """
    Applies SpecAugment style time and frequency masking to a log-mel spectrogram.
    Input shape: (N_MELS, T)
    """
    augmented = log_mel_spec.copy()
    n_mels, n_steps = augmented.shape
    
    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        if f > 0:
            f0 = random.randint(0, n_mels - f)
            augmented[f0:f0+f, :] = 0
            
    # Time masking
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        if t > 0 and t < n_steps:
            t0 = random.randint(0, n_steps - t)
            augmented[:, t0:t0+t] = 0
            
    return augmented
