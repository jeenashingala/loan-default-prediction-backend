import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime
from app.database import Base

class PredictionRecord(Base):
    """SQLAlchemy model representing a saved loan default prediction in the database."""
    
    __tablename__ = "predictions"

    id = Column(String(50), primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    applicant_name = Column(String(100), nullable=True)
    
    # 16 Loan Application Features
    age = Column(Float, nullable=False)
    income = Column(Float, nullable=False)
    loan_amount = Column(Float, nullable=False)
    credit_score = Column(Float, nullable=False)
    months_employed = Column(Float, nullable=False)
    num_credit_lines = Column(Float, nullable=False)
    interest_rate = Column(Float, nullable=False)
    loan_term = Column(Float, nullable=False)
    dti_ratio = Column(Float, nullable=False)
    education = Column(String(50), nullable=False)
    employment_type = Column(String(50), nullable=False)
    marital_status = Column(String(50), nullable=False)
    has_mortgage = Column(String(10), nullable=False)
    has_dependents = Column(String(10), nullable=False)
    loan_purpose = Column(String(50), nullable=False)
    has_co_signer = Column(String(10), nullable=False)
    
    # Model Predictions & Telemetry
    prediction = Column(Integer, nullable=False)
    prediction_label = Column(String(50), nullable=False)
    default_probability = Column(Float, nullable=False)
    no_default_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    model_name = Column(String(100), nullable=False)
