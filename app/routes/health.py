from fastapi import APIRouter
from app.model import ml_container
from app.config import settings
from app.schemas import HealthResponse

router = APIRouter(tags=["Health & Status"])

@router.get("/", summary="Root API endpoint")
def root():
    """Returns basic heartbeat confirmation."""
    return {"message": "Loan Default Prediction API is running"}

@router.get("/health", response_model=HealthResponse, summary="API and Model Health Check")
def health_check():
    """Reports API status, model loading status, and active model identifier."""
    is_loaded = ml_container.is_loaded()
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model=settings.MODEL_NAME
    )
