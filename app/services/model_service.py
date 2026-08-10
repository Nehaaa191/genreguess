import json
import torch
import torch.nn.functional as F
from app.config import MODEL_WEIGHTS_PATH, LABEL_MAP_PATH, MODEL_CONFIG_PATH
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.deep_learning.model import GenreCNN

class ModelService:
    def __init__(self):
        self.model = None
        self.label_map = None
        
    def load(self):
        with open(LABEL_MAP_PATH, 'r') as f:
            self.label_map = json.load(f)
            
        with open(MODEL_CONFIG_PATH, 'r') as f:
            model_config = json.load(f)
            
        self.model = GenreCNN(**model_config)
        
        # Load weights on CPU for inference
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
        self.model.eval()
        
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, tensor: torch.Tensor):
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded.")
            
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().numpy()
            
        max_idx = probs.argmax().item()
        genre = self.label_map[str(max_idx)]
        confidence = float(probs[max_idx])
        
        prob_dict = {self.label_map[str(i)]: float(p) for i, p in enumerate(probs)}
        
        return genre, confidence, prob_dict

model_service = ModelService()
