import os
import sys
import torch
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.deep_learning.model import GenreCNN

def generate_dummy_artifacts():
    os.makedirs('artifacts', exist_ok=True)
    
    # 1. Model config
    model_config = {"num_classes": 10, "dropout_rate": 0.4}
    with open('artifacts/model_config.json', 'w') as f:
        json.dump(model_config, f)
        
    # 2. Preprocessing config
    prep_config = {
        "sample_rate": 22050,
        "duration": 29,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "power": 2.0
    }
    with open('artifacts/preprocessing_config.json', 'w') as f:
        json.dump(prep_config, f)
        
    # 3. Label Map
    GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    INV_LABEL_MAP = {str(i): g for i, g in enumerate(GENRES)}
    with open('artifacts/label_map.json', 'w') as f:
        json.dump(INV_LABEL_MAP, f)
        
    # 4. Dummy Model Weights
    model = GenreCNN(num_classes=10)
    torch.save(model.state_dict(), 'artifacts/genre_cnn.pt')
    
    print("Dummy artifacts generated successfully!")

if __name__ == "__main__":
    generate_dummy_artifacts()
