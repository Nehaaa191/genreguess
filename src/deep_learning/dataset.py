import os
import torch
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.deep_learning.augmentations import spec_augment

class GTZANSpectrogramDataset(Dataset):
    def __init__(self, file_list, label_map, split, precompute_dir, augment=False):
        """
        file_list: list of relative paths (e.g., 'blues/blues.00000.wav')
        label_map: dict mapping label_name to int ID (e.g., {'blues': 0})
        split: 'train', 'val', or 'test'
        precompute_dir: path to the precomputed .npy spectrograms directory
        augment: bool, whether to apply SpecAugment (only if split == 'train')
        """
        self.file_list = file_list
        self.label_map = label_map
        self.split = split
        self.precompute_dir = precompute_dir
        self.augment = augment and split == 'train'
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        rel_path = self.file_list[idx]
        genre_name = os.path.dirname(rel_path)
        filename = os.path.basename(rel_path)
        npy_filename = filename.replace(".wav", ".npy")
        
        npy_path = os.path.join(self.precompute_dir, genre_name, npy_filename)
        
        try:
            spec = np.load(npy_path)
        except Exception:
            # Fallback if corrupted or missing: zero tensor of expected shape
            spec = np.zeros((128, 1250), dtype=np.float32)
            
        # Enforce exact temporal dimension to avoid dataloader collation errors
        TARGET_FRAMES = 1250
        if spec.shape[1] > TARGET_FRAMES:
            spec = spec[:, :TARGET_FRAMES]
        elif spec.shape[1] < TARGET_FRAMES:
            pad_width = TARGET_FRAMES - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
            
        if self.augment:
            spec = spec_augment(spec)
            
        # Add channel dimension: (1, 128, T)
        spec = np.expand_dims(spec, axis=0)
        tensor = torch.tensor(spec, dtype=torch.float32)
        
        label_idx = self.label_map[genre_name]
        return tensor, label_idx
