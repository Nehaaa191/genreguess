import os
import sys
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.common.split_utils import load_splits
from src.deep_learning.dataset import GTZANSpectrogramDataset
from src.deep_learning.model import GenreCNN

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
LABEL_MAP = {g: i for i, g in enumerate(GENRES)}

def test_speed():
    print("Testing train speed...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    precompute_dir = "data/processed/spectrograms"
    split_path = "data/splits/split_seed42.json"
    splits = load_splits(split_path)
    
    train_ds = GTZANSpectrogramDataset(splits['train'], LABEL_MAP, 'train', precompute_dir, augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    model = GenreCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}/22 - Time so far: {time.time() - start_time:.2f}s", flush=True)
    
    print(f"Epoch finished in {time.time() - start_time:.2f}s", flush=True)

if __name__ == "__main__":
    test_speed()
