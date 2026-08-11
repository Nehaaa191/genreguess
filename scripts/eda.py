import os
import sys
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def generate_eda_reports(raw_dir='data/raw/genres_original', reports_dir='reports/eda'):
    os.makedirs(reports_dir, exist_ok=True)
    
    genres = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    # 1. Class Distribution
    track_counts = {}
    total_tracks = 0
    
    print("Collecting dataset statistics...")
    for g in genres:
        genre_path = os.path.join(raw_dir, g)
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        track_counts[g] = len(files)
        total_tracks += len(files)
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(track_counts.keys()), y=list(track_counts.values()), palette='viridis')
    plt.title(f'GTZAN Class Distribution (Total: {total_tracks} tracks)')
    plt.ylabel('Number of Tracks')
    plt.xlabel('Genre')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'class_distribution.png'))
    plt.close()
    
    # 2. Representative Waveform and Spectrogram
    sample_genre = genres[0]
    sample_genre_path = os.path.join(raw_dir, sample_genre)
    sample_file = [f for f in os.listdir(sample_genre_path) if f.endswith('.wav')][0]
    sample_path = os.path.join(sample_genre_path, sample_file)
    
    print(f"Loading representative sample: {sample_path}")
    y, sr = librosa.load(sample_path, sr=22050)
    
    # Waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title(f'Waveform - {sample_genre.capitalize()} ({sample_file})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'representative_waveform.png'))
    plt.close()
    
    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram - {sample_genre.capitalize()}')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'representative_spectrogram.png'))
    plt.close()

    # Save summary stats
    stats = {
        "total_tracks": total_tracks,
        "classes": len(genres),
        "sample_rate": sr,
        "track_duration_seconds": len(y) / sr,
        "track_counts": track_counts
    }
    with open(os.path.join(reports_dir, 'eda_summary.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"EDA complete. Plots saved to {reports_dir}")

if __name__ == '__main__':
    generate_eda_reports()
