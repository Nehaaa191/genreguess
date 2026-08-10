import os
import json
import hashlib
from collections import defaultdict
from sklearn.model_selection import train_test_split

def _hash_file(filepath: str) -> str:
    """Returns SHA256 hash of a file to check for exact duplicates."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception:
        return ""

def create_and_save_splits(data_dir: str, output_split_path: str, seed: int = 42):
    """
    Finds all .wav files in data_dir, creates a stratified 70/15/15 split,
    checks for exact file duplicates to avoid leakage, and saves the split to output_split_path.
    """
    all_files = []
    labels = []
    
    genres = sorted(os.listdir(data_dir))
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        if os.path.isdir(genre_dir):
            for filename in os.listdir(genre_dir):
                if filename.endswith(".wav"):
                    filepath = os.path.join(genre, filename)
                    all_files.append(filepath)
                    labels.append(genre)
                    
    # Leakage check: verify no exact file duplicates
    print("Checking for exact file duplicates (leakage check)...")
    hash_to_files = defaultdict(list)
    for filepath in all_files:
        full_path = os.path.join(data_dir, filepath)
        h = _hash_file(full_path)
        if h:
            hash_to_files[h].append(filepath)
            
    duplicates = {k: v for k, v in hash_to_files.items() if len(v) > 1}
    if duplicates:
        print(f"WARNING: Found {len(duplicates)} exact duplicates. Documenting leakage.")
    else:
        print("No exact file duplicates found via SHA256 hashing.")

    # Note: GTZAN is known to have some near-duplicates (same song slightly different segment).
    # A true robust leakage check would involve audio fingerprinting, but for this project,
    # we explicitly document this known limitation.

    # Split 70% train, 30% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_files, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    
    # Split temp into 15% val, 15% test (which is 50% of the 30% temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    
    split_dict = {
        "train": X_train,
        "val": X_val,
        "test": X_test
    }
    
    os.makedirs(os.path.dirname(output_split_path), exist_ok=True)
    with open(output_split_path, 'w') as f:
        json.dump(split_dict, f, indent=4)
        
    print(f"Split created and saved to {output_split_path}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

def load_splits(split_path: str):
    """Loads the split dictionary."""
    with open(split_path, 'r') as f:
        return json.load(f)
