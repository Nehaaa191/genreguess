import os
import sys
import numpy as np

# Add the project root to sys.path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.common.audio_preprocessing import process_audio_file
from src.common.split_utils import create_and_save_splits

def main():
    raw_data_dir = os.path.abspath(os.path.join("data", "raw", "genres_original"))
    output_dir = os.path.abspath(os.path.join("data", "processed", "spectrograms"))
    split_path = os.path.abspath(os.path.join("data", "splits", "split_seed42.json"))
    
    # 1. Create Train/Val/Test Splits
    if not os.path.exists(split_path):
        print("Creating dataset splits...")
        create_and_save_splits(raw_data_dir, split_path)
    else:
        print(f"Splits already exist at {split_path}")
        
    # 2. Precompute Spectrograms
    os.makedirs(output_dir, exist_ok=True)
    
    genres = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
    
    count = 0
    print("Precomputing spectrograms...")
    for genre in genres:
        genre_dir = os.path.join(raw_data_dir, genre)
        out_genre_dir = os.path.join(output_dir, genre)
        os.makedirs(out_genre_dir, exist_ok=True)
        
        for filename in os.listdir(genre_dir):
            if filename.endswith(".wav"):
                wav_path = os.path.join(genre_dir, filename)
                npy_filename = filename.replace(".wav", ".npy")
                npy_path = os.path.join(out_genre_dir, npy_filename)
                
                if not os.path.exists(npy_path):
                    try:
                        spec = process_audio_file(wav_path)
                        np.save(npy_path, spec)
                        count += 1
                    except Exception as e:
                        print(f"Failed to process {wav_path}: {e}")
                
                if count > 0 and count % 50 == 0:
                    print(f"Processed {count} files...")
                    
    print(f"Done. Processed {count} new files. Precomputed spectrograms are in {output_dir}")

if __name__ == "__main__":
    main()
