from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app
from app.services.model_service import model_service

client = TestClient(app)

def test_health_endpoint(monkeypatch):
    # Mock model loaded to be true
    monkeypatch.setattr(model_service, "is_loaded", lambda: True)
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] == True
