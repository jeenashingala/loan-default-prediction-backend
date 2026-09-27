from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    DashboardResponse,
    AnalyticsMetricsResponse,
    ModelInfoResponse
)
from app.services.analytics_service import (
    get_dashboard_data,
    get_analytics_metrics,
    get_model_info
)

router = APIRouter(tags=["Analytics & Model Intelligence"])

@router.get(
    "/analytics/dashboard",
    response_model=DashboardResponse,
    summary="Dashboard Overview Metrics",
    description="Calculates real application totals, default rate, risk distribution, and 7-day trend from the database."
)
def get_dashboard_analytics(db: Session = Depends(get_db)):
    return get_dashboard_data(db)

@router.get(
    "/analytics/metrics",
    response_model=AnalyticsMetricsResponse,
    summary="Detailed Analytics & Segmentation",
    description="Aggregate cohort analytics across employment type, purpose, risk distribution, and scatter plot points."
)
def get_detailed_analytics(db: Session = Depends(get_db)):
    return get_analytics_metrics(db)

@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Model Architecture & Metadata",
    description="Returns metadata about the active ML model, scaler, and 24 training features."
)
def get_model_metadata():
    return get_model_info()
