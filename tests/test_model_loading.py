import pytest
import torch
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.deep_learning.model import GenreCNN

def test_model_loading(tmp_path):
    # Create dummy model and save state dict
    model = GenreCNN(num_classes=10)
    torch.save(model.state_dict(), tmp_path / "dummy_model.pt")
    
    # Save dummy configs
    with open(tmp_path / "model_config.json", "w") as f:
        json.dump({"num_classes": 10, "dropout_rate": 0.4}, f)
        
    # Load model
    with open(tmp_path / "model_config.json", "r") as f:
        config = json.load(f)
        
    loaded_model = GenreCNN(**config)
    loaded_model.load_state_dict(torch.load(tmp_path / "dummy_model.pt"))
    
    # Forward pass with dummy tensor
    dummy_input = torch.randn(1, 1, 128, 1250)
    output = loaded_model(dummy_input)
    
    assert output.shape == (1, 10)
