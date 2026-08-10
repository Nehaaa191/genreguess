import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app
from app.services.model_service import model_service
from src.common.audio_preprocessing import SAMPLE_RATE

client = TestClient(app)

@pytest.fixture
def synthetic_audio_path(tmp_path):
    duration = 2
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    path = tmp_path / "synthetic.wav"
    sf.write(path, audio, SAMPLE_RATE)
    return str(path)

def test_predict_endpoint(synthetic_audio_path, monkeypatch):
    # Mock model predict
    def mock_predict(tensor):
        return "blues", 0.95, {"blues": 0.95, "rock": 0.05}
        
    monkeypatch.setattr(model_service, "predict", mock_predict)
    monkeypatch.setattr(model_service, "is_loaded", lambda: True)
    
    # Needs a config file since process_audio_bytes reads it
    os.makedirs('artifacts', exist_ok=True)
    import json
    with open('artifacts/preprocessing_config.json', 'w') as f:
        json.dump({
            "sample_rate": 22050,
            "duration": 29,
            "n_fft": 2048,
            "hop_length": 512,
            "n_mels": 128,
            "power": 2.0
        }, f)
        
    with open(synthetic_audio_path, "rb") as f:
        response = client.post("/predict", files={"file": ("test.wav", f, "audio/wav")})
        
    assert response.status_code == 200
    data = response.json()
    assert data["genre"] == "blues"
    assert data["confidence"] == 0.95
    assert sum(data["probabilities"].values()) == pytest.approx(1.0)
    
def test_predict_endpoint_invalid_file():
    # Test uploading a text file
    response = client.post(
        "/predict", 
        files={"file": ("test.txt", b"not audio data", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only .wav files are supported" in response.json()["detail"]
