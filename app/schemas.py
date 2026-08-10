from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    genre: str
    confidence: float
    probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class ErrorResponse(BaseModel):
    detail: str
