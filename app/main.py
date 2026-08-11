import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.routers import health, predict
from app.services.model_service import model_service

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Music Genre Classification API",
    description="API for predicting music genre from audio files.",
    version="1.0.0"
)

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
        "http://localhost:5175", "http://127.0.0.1:5175"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request parameters."}
    )

# Lifespan / Startup Event
@app.on_event("startup")
async def startup_event():
    try:
        model_service.load()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")

app.include_router(health.router)
app.include_router(predict.router)

