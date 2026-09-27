from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

class LoanPredictionRequest(BaseModel):
    """Pydantic model validating incoming applicant loan parameters."""
    applicantName: Optional[str] = Field(None, description="Optional full name of loan applicant")
    Age: float = Field(..., ge=18, le=100, description="Applicant age in solar years (18-100)")
    Income: float = Field(..., gt=0, description="Annual gross verified income in currency units")
    LoanAmount: float = Field(..., gt=0, description="Total requested principal loan amount")
    CreditScore: float = Field(..., ge=300, le=850, description="Credit Bureau Score (300-850)")
    MonthsEmployed: float = Field(..., ge=0, description="Tenure with employer in completed months")
    NumCreditLines: float = Field(..., ge=0, description="Number of active revolving credit facilities")
    InterestRate: float = Field(..., ge=0.0, le=100.0, description="Contractual loan interest rate APR (%)")
    LoanTerm: float = Field(..., ge=1, le=360, description="Repayment duration in months")
    DTIRatio: float = Field(..., ge=0.0, le=2.0, description="Debt-to-Income leverage ratio (0.0 - 2.0)")
    Education: Literal["High School", "Bachelor's", "Master's", "PhD"]
    EmploymentType: Literal["Full-time", "Part-time", "Self-employed", "Unemployed"]
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    HasMortgage: Literal["Yes", "No"]
    HasDependents: Literal["Yes", "No"]
    LoanPurpose: Literal["Auto", "Business", "Education", "Home", "Other"]
    HasCoSigner: Literal["Yes", "No"]

class LoanPredictionResponse(BaseModel):
    """Standard response model for real-time model inference."""
    id: str
    prediction: int = Field(..., description="0 for No Default, 1 for Default")
    prediction_label: str = Field(..., description="'No Default' or 'Default'")
    probability: float = Field(..., description="Probability of default (0.0 to 1.0)")
    default_probability: float = Field(..., description="Calibrated default probability")
    no_default_probability: float = Field(..., description="Complementary repayment probability")
    risk_level: str = Field(..., description="'Low', 'Medium', or 'High'")
    model: str = Field(..., description="Trained model identifier")
    created_at: Optional[str] = None
    applicantName: Optional[str] = None

class PredictionHistoryItem(BaseModel):
    """Prediction history entry representation."""
    id: str
    applicantName: Optional[str] = None
    date: str
    created_at: str
    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str
    prediction: int
    prediction_label: str
    probability: float
    default_probability: float
    no_default_probability: float
    risk_level: str
    model: str

class PredictionHistoryResponse(BaseModel):
    """Paginated prediction history response."""
    items: List[PredictionHistoryItem]
    total: int
    page: int
    limit: int
    totalPages: int

class DashboardStatsSummary(BaseModel):
    totalApplications: int
    totalApplicationsLabel: str
    defaultPredictions: int
    defaultPredictionsLabel: str
    defaultRate: float
    defaultRateLabel: str
    modelName: str
    modelLabel: str

class DistributionItem(BaseModel):
    name: str
    value: int
    percentage: float
    color: str

class ApplicationsOverTimeItem(BaseModel):
    day: str
    applications: int
    defaults: int

class DashboardResponse(BaseModel):
    """Aggregated dashboard telemetry metrics derived from database."""
    total_applications: int
    total_predictions: int
    default_predictions: int
    non_default_predictions: int
    default_rate: float
    high_risk_predictions: int
    medium_risk_predictions: int
    low_risk_predictions: int
    stats: DashboardStatsSummary
    distribution: List[DistributionItem]
    applicationsOverTime: List[ApplicationsOverTimeItem]

class CohortRiskItem(BaseModel):
    employment: Optional[str] = None
    purpose: Optional[str] = None
    total: int
    defaults: int
    defaultRate: float

class ScatterItem(BaseModel):
    creditScore: float
    probability: float
    applicant: str

class ModelEvaluationMetrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1Score: Optional[float] = None
    rocAuc: Optional[float] = None

class FeatureMetaItem(BaseModel):
    name: str
    type: str
    range: str
    category: str
    description: str

class WorkflowStepItem(BaseModel):
    step: int
    title: str
    description: str

class ModelMetaInfo(BaseModel):
    modelName: str
    algorithm: str
    targetVariable: str
    problemType: str
    class_weight: str
    features_count: int
    scaler: str
    metrics: ModelEvaluationMetrics
    features: List[FeatureMetaItem]
    workflowSteps: List[WorkflowStepItem]

class AnalyticsMetricsResponse(BaseModel):
    """Aggregate analytics for portfolio segmentation and credit scoring patterns."""
    modelMeta: ModelMetaInfo
    distribution: List[DistributionItem]
    byEmployment: List[CohortRiskItem]
    byPurpose: List[CohortRiskItem]
    scatterData: List[ScatterItem]

class ModelInfoResponse(BaseModel):
    """Dedicated model metadata for the About Model page."""
    name: str
    type: str
    class_weight: str
    features: int
    scaler: str
    training_features: List[str]
    modelMeta: Optional[ModelMetaInfo] = None

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    model: str
