from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    LoanPredictionRequest,
    LoanPredictionResponse,
    PredictionHistoryResponse,
    PredictionHistoryItem
)
from app.services.prediction_service import (
    run_prediction_and_save,
    get_predictions_history,
    get_prediction_by_id,
    delete_prediction_by_id
)

router = APIRouter(tags=["Loan Predictions"])

@router.post(
    "/predict",
    response_model=LoanPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Loan Default Prediction",
    description="Takes applicant parameters, executes preprocessing and Balanced Logistic Regression inference, stores result to database, and returns probability assessment."
)
def predict_loan(
    request: LoanPredictionRequest,
    db: Session = Depends(get_db)
):
    try:
        response = run_prediction_and_save(request, db)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get(
    "/predictions",
    response_model=PredictionHistoryResponse,
    summary="Get Prediction History",
    description="Returns stored loan predictions from database with optional filtering, sorting, and pagination."
)
def get_predictions(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Page size limit"),
    search: Optional[str] = Query(None, description="Search query by ID or applicant name"),
    risk: Optional[str] = Query("all", description="Risk level filter: all, low, medium, high"),
    sortBy: Optional[str] = Query("date_desc", description="Sort criteria"),
    db: Session = Depends(get_db)
):
    return get_predictions_history(
        db=db,
        page=page,
        limit=limit,
        search=search,
        risk=risk,
        sort_by=sortBy
    )

@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionHistoryItem,
    summary="Get Prediction Details by ID",
    description="Fetch single prediction record and applicant attributes."
)
def get_prediction(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    record = get_prediction_by_id(db, prediction_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID '{prediction_id}' was not found."
        )
    return record

@router.delete(
    "/predictions/{prediction_id}",
    summary="Delete Prediction Record",
    description="Deletes a prediction record from the database by ID."
)
def delete_prediction(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    success = delete_prediction_by_id(db, prediction_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID '{prediction_id}' was not found."
        )
    return {"success": True, "id": prediction_id, "message": "Record successfully deleted"}
