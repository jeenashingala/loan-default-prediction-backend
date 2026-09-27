import uuid
import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc

from app.model import ml_container
from app.preprocessing import preprocess_features, calculate_risk_level
from app.models.prediction import PredictionRecord
from app.schemas import (
    LoanPredictionRequest,
    LoanPredictionResponse,
    PredictionHistoryResponse,
    PredictionHistoryItem
)
from app.config import settings

def generate_application_id() -> str:
    """Generate a clean, professional application identifier e.g., APP-7821."""
    short_code = uuid.uuid4().hex[:5].upper()
    return f"APP-{short_code}"

def run_prediction_and_save(
    request: LoanPredictionRequest,
    db: Session
) -> LoanPredictionResponse:
    """
    Executes model inference on validated input parameters and persists the result to SQLite.
    """
    model = ml_container.get_model()
    scaler = ml_container.get_scaler()
    feature_columns = ml_container.get_feature_columns()

    # 1. Exact preprocessing and scaling
    scaled_array = preprocess_features(request, feature_columns, scaler)

    # 2. Model inference via trained Balanced Logistic Regression
    raw_pred = int(model.predict(scaled_array)[0])
    raw_proba = model.predict_proba(scaled_array)[0]

    # Classes are [0, 1] -> 0: No Default, 1: Default
    no_default_prob = round(float(raw_proba[0]), 4)
    default_prob = round(float(raw_proba[1]), 4)

    prediction_label = "Default" if raw_pred == 1 else "No Default"
    risk_level = calculate_risk_level(default_prob)
    app_id = generate_application_id()
    now = datetime.datetime.utcnow()

    # 3. Save into SQLite database
    record = PredictionRecord(
        id=app_id,
        created_at=now,
        applicant_name=request.applicantName or f"Applicant {app_id.replace('APP-', '')}",
        age=request.Age,
        income=request.Income,
        loan_amount=request.LoanAmount,
        credit_score=request.CreditScore,
        months_employed=request.MonthsEmployed,
        num_credit_lines=request.NumCreditLines,
        interest_rate=request.InterestRate,
        loan_term=request.LoanTerm,
        dti_ratio=request.DTIRatio,
        education=request.Education,
        employment_type=request.EmploymentType,
        marital_status=request.MaritalStatus,
        has_mortgage=request.HasMortgage,
        has_dependents=request.HasDependents,
        loan_purpose=request.LoanPurpose,
        has_co_signer=request.HasCoSigner,
        prediction=raw_pred,
        prediction_label=prediction_label,
        default_probability=default_prob,
        no_default_probability=no_default_prob,
        risk_level=risk_level,
        model_name=settings.MODEL_NAME,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return LoanPredictionResponse(
        id=record.id,
        prediction=record.prediction,
        prediction_label=record.prediction_label,
        probability=record.default_probability,
        default_probability=record.default_probability,
        no_default_probability=record.no_default_probability,
        risk_level=record.risk_level,
        model=record.model_name,
        created_at=record.created_at.isoformat(),
        applicantName=record.applicant_name,
    )

def get_predictions_history(
    db: Session,
    page: int = 1,
    limit: int = 10,
    search: Optional[str] = None,
    risk: Optional[str] = "all",
    sort_by: Optional[str] = "date_desc"
) -> PredictionHistoryResponse:
    """Retrieves paginated prediction records with filtering and sorting."""
    query = db.query(PredictionRecord)

    # Risk level filter
    if risk and risk.lower() != "all":
        query = query.filter(PredictionRecord.risk_level.ilike(f"%{risk.strip()}%"))

    # Search filter (ID, applicant name, or loan purpose)
    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(
            (PredictionRecord.id.ilike(term)) |
            (PredictionRecord.applicant_name.ilike(term)) |
            (PredictionRecord.loan_purpose.ilike(term))
        )

    # Sorting
    if sort_by == "date_asc":
        query = query.order_by(asc(PredictionRecord.created_at))
    elif sort_by == "prob_desc":
        query = query.order_by(desc(PredictionRecord.default_probability))
    elif sort_by == "prob_asc":
        query = query.order_by(asc(PredictionRecord.default_probability))
    elif sort_by == "amount_desc":
        query = query.order_by(desc(PredictionRecord.loan_amount))
    elif sort_by == "score_desc":
        query = query.order_by(desc(PredictionRecord.credit_score))
    else:  # default 'date_desc'
        query = query.order_by(desc(PredictionRecord.created_at))

    total = query.count()
    offset = (page - 1) * limit
    records = query.offset(offset).limit(limit).all()

    items = [
        PredictionHistoryItem(
            id=r.id,
            applicantName=r.applicant_name,
            date=r.created_at.isoformat(),
            created_at=r.created_at.isoformat(),
            Age=r.age,
            Income=r.income,
            LoanAmount=r.loan_amount,
            CreditScore=r.credit_score,
            MonthsEmployed=r.months_employed,
            NumCreditLines=r.num_credit_lines,
            InterestRate=r.interest_rate,
            LoanTerm=r.loan_term,
            DTIRatio=r.dti_ratio,
            Education=r.education,
            EmploymentType=r.employment_type,
            MaritalStatus=r.marital_status,
            HasMortgage=r.has_mortgage,
            HasDependents=r.has_dependents,
            LoanPurpose=r.loan_purpose,
            HasCoSigner=r.has_co_signer,
            prediction=r.prediction,
            prediction_label=r.prediction_label,
            probability=r.default_probability,
            default_probability=r.default_probability,
            no_default_probability=r.no_default_probability,
            risk_level=r.risk_level,
            model=r.model_name,
        )
        for r in records
    ]

    total_pages = max(1, (total + limit - 1) // limit) if total > 0 else 1

    return PredictionHistoryResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        totalPages=total_pages
    )

def get_prediction_by_id(db: Session, prediction_id: str) -> Optional[PredictionHistoryItem]:
    """Fetch single prediction record by ID."""
    r = db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    if not r:
        return None
    return PredictionHistoryItem(
        id=r.id,
        applicantName=r.applicant_name,
        date=r.created_at.isoformat(),
        created_at=r.created_at.isoformat(),
        Age=r.age,
        Income=r.income,
        LoanAmount=r.loan_amount,
        CreditScore=r.credit_score,
        MonthsEmployed=r.months_employed,
        NumCreditLines=r.num_credit_lines,
        InterestRate=r.interest_rate,
        LoanTerm=r.loan_term,
        DTIRatio=r.dti_ratio,
        Education=r.education,
        EmploymentType=r.employment_type,
        MaritalStatus=r.marital_status,
        HasMortgage=r.has_mortgage,
        HasDependents=r.has_dependents,
        LoanPurpose=r.loan_purpose,
        HasCoSigner=r.has_co_signer,
        prediction=r.prediction,
        prediction_label=r.prediction_label,
        probability=r.default_probability,
        default_probability=r.default_probability,
        no_default_probability=r.no_default_probability,
        risk_level=r.risk_level,
        model=r.model_name,
    )

def delete_prediction_by_id(db: Session, prediction_id: str) -> bool:
    """Delete prediction record by ID from database."""
    r = db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    if not r:
        return False
    db.delete(r)
    db.commit()
    return True
