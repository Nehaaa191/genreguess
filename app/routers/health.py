from fastapi import APIRouter
from app.schemas import HealthResponse
from app.services.model_service import model_service

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded()
    )
