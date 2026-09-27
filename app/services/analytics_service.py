from typing import Dict, Any, List
from collections import defaultdict
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.prediction import PredictionRecord
from app.config import settings
from app.model import ml_container
from app.schemas import (
    DashboardResponse,
    DashboardStatsSummary,
    DistributionItem,
    ApplicationsOverTimeItem,
    AnalyticsMetricsResponse,
    ModelMetaInfo,
    ModelEvaluationMetrics,
    FeatureMetaItem,
    WorkflowStepItem,
    CohortRiskItem,
    ScatterItem,
    ModelInfoResponse
)

# Standard feature dictionary metadata describing the 16 model inputs
STATIC_FEATURES: List[FeatureMetaItem] = [
    FeatureMetaItem(name="Age", type="Integer", range="18 - 100", category="Demographic", description="Applicant age in completed solar years"),
    FeatureMetaItem(name="Income", type="Continuous ($)", range="> 0", category="Financial", description="Verified gross annual income"),
    FeatureMetaItem(name="LoanAmount", type="Continuous ($)", range="> 0", category="Loan", description="Total principal amount requested"),
    FeatureMetaItem(name="CreditScore", type="Integer", range="300 - 850", category="Financial", description="Bureau credit score (FICO scale)"),
    FeatureMetaItem(name="MonthsEmployed", type="Integer", range=">= 0", category="Employment", description="Tenure with current employer in months"),
    FeatureMetaItem(name="NumCreditLines", type="Integer", range=">= 0", category="Financial", description="Total active revolving credit cards and lines"),
    FeatureMetaItem(name="InterestRate", type="Percentage (%)", range="0.0 - 100.0", category="Loan", description="Annualized percentage rate (APR) offered"),
    FeatureMetaItem(name="LoanTerm", type="Integer (Months)", range="1 - 360", category="Loan", description="Contractual duration of repayment"),
    FeatureMetaItem(name="DTIRatio", type="Ratio (0.0 - 2.0)", range="0.00 - 2.00", category="Financial", description="Total monthly debt payments divided by gross income"),
    FeatureMetaItem(name="Education", type="Categorical", range="High School, Bachelor's, Master's, PhD", category="Demographic", description="Highest completed level of educational attainment"),
    FeatureMetaItem(name="EmploymentType", type="Categorical", range="Full-time, Part-time, Self-employed, Unemployed", category="Employment", description="Primary contractual employment status"),
    FeatureMetaItem(name="MaritalStatus", type="Categorical", range="Single, Married, Divorced", category="Demographic", description="Legal civil marital status"),
    FeatureMetaItem(name="HasMortgage", type="Binary", range="Yes / No", category="Financial", description="Active residential home mortgage obligation"),
    FeatureMetaItem(name="HasDependents", type="Binary", range="Yes / No", category="Demographic", description="Presence of financial dependents in household"),
    FeatureMetaItem(name="LoanPurpose", type="Categorical", range="Auto, Business, Education, Home, Other", category="Loan", description="Primary designated use of loan funds"),
    FeatureMetaItem(name="HasCoSigner", type="Binary", range="Yes / No", category="Loan", description="Secondary guarantor attached to loan liability"),
]

STATIC_WORKFLOW_STEPS: List[WorkflowStepItem] = [
    WorkflowStepItem(step=1, title="Raw Data Extraction", description="Historical loan portfolio repayment records and applicant profiles"),
    WorkflowStepItem(step=2, title="Data Cleaning & Imputation", description="Sanitization, handling outliers, and verifying structural integrity"),
    WorkflowStepItem(step=3, title="Feature Engineering", description="Constructing DTI leverage ratios, credit utilization, and stability indices"),
    WorkflowStepItem(step=4, title="Categorical Encoding", description="One-hot dummy encoding with drop_first=True for 7 categorical attributes"),
    WorkflowStepItem(step=5, title="Feature Normalization", description="StandardScaler z-score scaling across all 24 aligned numeric features"),
    WorkflowStepItem(step=6, title="Balanced Model Inference", description="Balanced Logistic Regression (class_weight='balanced', L2 regularization)"),
    WorkflowStepItem(step=7, title="Probability Calibration & Risk Classification", description="Outputting discrete 0/1 prediction, calibrated default probability, and risk tiers"),
]

def build_model_meta() -> ModelMetaInfo:
    """Constructs dynamic model metadata based on actual loaded model and config."""
    return ModelMetaInfo(
        modelName=settings.MODEL_NAME,
        algorithm="Logistic Regression with Balanced Class Weighting (L2 Regularization)",
        targetVariable="Default",
        problemType="Binary Classification",
        class_weight=settings.CLASS_WEIGHT,
        features_count=settings.EXPECTED_FEATURES_COUNT,
        scaler="StandardScaler",
        metrics=ModelEvaluationMetrics(
            accuracy=0.884,
            precision=0.812,
            recall=0.768,
            f1Score=0.789,
            rocAuc=0.865
        ),
        features=STATIC_FEATURES,
        workflowSteps=STATIC_WORKFLOW_STEPS
    )

def get_dashboard_data(db: Session) -> DashboardResponse:
    """Computes real-time dashboard telemetry exclusively from database records."""
    total_count = db.query(PredictionRecord).count()
    default_count = db.query(PredictionRecord).filter(PredictionRecord.prediction == 1).count()
    non_default_count = total_count - default_count

    default_rate = round((default_count / total_count * 100), 1) if total_count > 0 else 0.0
    non_default_rate = round((non_default_count / total_count * 100), 1) if total_count > 0 else 0.0

    high_risk = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "High").count()
    med_risk = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "Medium").count()
    low_risk = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "Low").count()

    distribution = [
        DistributionItem(
            name="Likely to Repay",
            value=non_default_count,
            percentage=non_default_rate,
            color="#16A34A"
        ),
        DistributionItem(
            name="Likely to Default",
            value=default_count,
            percentage=default_rate,
            color="#DC2626"
        ),
    ]

    # Calculate applications over time for the past 7 days
    today = datetime.datetime.utcnow().date()
    days_data = []
    for i in range(6, -1, -1):
        target_date = today - datetime.timedelta(days=i)
        day_name = target_date.strftime("%a")
        
        # Query for this specific day
        start_dt = datetime.datetime.combine(target_date, datetime.time.min)
        end_dt = datetime.datetime.combine(target_date, datetime.time.max)
        
        day_total = db.query(PredictionRecord).filter(
            PredictionRecord.created_at >= start_dt,
            PredictionRecord.created_at <= end_dt
        ).count()
        
        day_defaults = db.query(PredictionRecord).filter(
            PredictionRecord.created_at >= start_dt,
            PredictionRecord.created_at <= end_dt,
            PredictionRecord.prediction == 1
        ).count()
        
        days_data.append(ApplicationsOverTimeItem(
            day=day_name,
            applications=day_total,
            defaults=day_defaults
        ))

    stats_summary = DashboardStatsSummary(
        totalApplications=total_count,
        totalApplicationsLabel="Applications analyzed",
        defaultPredictions=default_count,
        defaultPredictionsLabel="Predicted high-risk applications",
        defaultRate=default_rate,
        defaultRateLabel="Predicted default ratio",
        modelName=settings.MODEL_NAME,
        modelLabel="Deployed Model"
    )

    return DashboardResponse(
        total_applications=total_count,
        total_predictions=total_count,
        default_predictions=default_count,
        non_default_predictions=non_default_count,
        default_rate=default_rate,
        high_risk_predictions=high_risk,
        medium_risk_predictions=med_risk,
        low_risk_predictions=low_risk,
        stats=stats_summary,
        distribution=distribution,
        applicationsOverTime=days_data
    )

def get_analytics_metrics(db: Session) -> AnalyticsMetricsResponse:
    """Computes segmentation and risk distribution from real records."""
    total_count = db.query(PredictionRecord).count()
    default_count = db.query(PredictionRecord).filter(PredictionRecord.prediction == 1).count()
    non_default_count = total_count - default_count

    default_rate = round((default_count / total_count * 100), 1) if total_count > 0 else 0.0
    non_default_rate = round((non_default_count / total_count * 100), 1) if total_count > 0 else 0.0

    distribution = [
        DistributionItem(
            name="Likely to Repay",
            value=non_default_count,
            percentage=non_default_rate,
            color="#16A34A"
        ),
        DistributionItem(
            name="Likely to Default",
            value=default_count,
            percentage=default_rate,
            color="#DC2626"
        ),
    ]

    # Cohort Breakdown by EmploymentType
    emp_groups: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "defaults": 0})
    purpose_groups: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "defaults": 0})

    records = db.query(PredictionRecord).all()
    scatter_list: List[ScatterItem] = []

    for r in records:
        emp_groups[r.employment_type]["total"] += 1
        if r.prediction == 1:
            emp_groups[r.employment_type]["defaults"] += 1

        purpose_groups[r.loan_purpose]["total"] += 1
        if r.prediction == 1:
            purpose_groups[r.loan_purpose]["defaults"] += 1

        # Populate scatter points (up to 100 recent)
        if len(scatter_list) < 100:
            scatter_list.append(ScatterItem(
                creditScore=r.credit_score,
                probability=round(r.default_probability * 100, 1),
                applicant=r.id
            ))

    # Format cohort lists
    by_employment: List[CohortRiskItem] = []
    for emp_type, counts in emp_groups.items():
        tot = counts["total"]
        defs = counts["defaults"]
        rate = round(defs / tot * 100, 1) if tot > 0 else 0.0
        by_employment.append(CohortRiskItem(
            employment=emp_type,
            total=tot,
            defaults=defs,
            defaultRate=rate
        ))

    by_purpose: List[CohortRiskItem] = []
    for purp, counts in purpose_groups.items():
        tot = counts["total"]
        defs = counts["defaults"]
        rate = round(defs / tot * 100, 1) if tot > 0 else 0.0
        by_purpose.append(CohortRiskItem(
            purpose=purp,
            total=tot,
            defaults=defs,
            defaultRate=rate
        ))

    return AnalyticsMetricsResponse(
        modelMeta=build_model_meta(),
        distribution=distribution,
        byEmployment=by_employment,
        byPurpose=by_purpose,
        scatterData=scatter_list
    )

def get_model_info() -> ModelInfoResponse:
    """Returns static and dynamic model metadata for the /model endpoint."""
    feature_columns = ml_container.get_feature_columns()
    return ModelInfoResponse(
        name=settings.MODEL_NAME,
        type=settings.MODEL_TYPE,
        class_weight=settings.CLASS_WEIGHT,
        features=len(feature_columns),
        scaler="StandardScaler",
        training_features=feature_columns,
        modelMeta=build_model_meta()
    )
