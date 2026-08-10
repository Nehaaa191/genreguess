from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import PredictionResponse, ErrorResponse
from app.services.model_service import model_service
from app.services.audio_service import process_audio_bytes

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict_genre(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
        
    try:
        file_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read file.")
        
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")
        
    try:
        tensor = process_audio_bytes(file_bytes)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing audio file.")
        
    try:
        genre, confidence, probs = model_service.predict(tensor)
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception:
        raise HTTPException(status_code=500, detail="Error during model inference.")
        
    return PredictionResponse(
        genre=genre,
        confidence=confidence,
        probabilities=probs
    )
