from typing import List, Dict, Any
import numpy as np
import pandas as pd
from app.schemas import LoanPredictionRequest

# Categorical and binary feature groups
NUMERICAL_COLUMNS = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio"
]

CATEGORICAL_COLUMNS = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "LoanPurpose"
]

BINARY_COLUMNS = [
    "HasMortgage",
    "HasDependents",
    "HasCoSigner"
]

# Combined categorical groups requiring one-hot dummy encoding
ALL_DUMMY_COLUMNS = CATEGORICAL_COLUMNS + BINARY_COLUMNS

def calculate_risk_level(default_probability: float) -> str:
    """
    Application-level UI risk tier interpretation of default probability.
    
    Thresholds:
      - [0.00, 0.33) -> Low Risk: Favorable credit profile, lower risk of charge-off.
      - [0.33, 0.66) -> Medium Risk: Mixed indicators, warrants underwriter review.
      - [0.66, 1.00] -> High Risk: Elevated leverage or credit strain.
      
    NOTE: This is strictly a presentation-layer risk categorization and does not
    constitute a binding statutory credit underwriting determination.
    """
    if default_probability < 0.33:
        return "Low"
    elif default_probability < 0.66:
        return "Medium"
    else:
        return "High"

def preprocess_features(
    request: LoanPredictionRequest,
    feature_columns: List[str],
    scaler: Any
) -> np.ndarray:
    """
    Reproduces training pipeline preprocessing for an incoming loan application:
      1. Converts request into a single-row pandas DataFrame.
      2. Applies one-hot encoding across categorical and binary attributes.
      3. Reindexes against feature_columns.pkl ensuring exactly 24 features in correct order.
      4. Fills any absent dummy columns with 0.
      5. Strips any unexpected columns.
      6. Applies StandardScaler transformation fitted during training.
      
    Returns:
      Scaled 2D numpy array of shape (1, 24).
    """
    raw_dict = {
        "Age": float(request.Age),
        "Income": float(request.Income),
        "LoanAmount": float(request.LoanAmount),
        "CreditScore": float(request.CreditScore),
        "MonthsEmployed": float(request.MonthsEmployed),
        "NumCreditLines": float(request.NumCreditLines),
        "InterestRate": float(request.InterestRate),
        "LoanTerm": float(request.LoanTerm),
        "DTIRatio": float(request.DTIRatio),
        "Education": str(request.Education),
        "EmploymentType": str(request.EmploymentType),
        "MaritalStatus": str(request.MaritalStatus),
        "HasMortgage": str(request.HasMortgage),
        "HasDependents": str(request.HasDependents),
        "LoanPurpose": str(request.LoanPurpose),
        "HasCoSigner": str(request.HasCoSigner),
    }

    df = pd.DataFrame([raw_dict])

    # One-hot encode categorical and binary variables
    df_encoded = pd.get_dummies(df, columns=ALL_DUMMY_COLUMNS, dtype=int)

    # Reindex to strictly align with feature_columns.pkl:
    # - Guarantees exact 24 column count
    # - Imputes 0 for dropped or absent reference categories
    # - Preserves exact training column order
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale the aligned 24 features using the pre-fitted StandardScaler
    scaled_array = scaler.transform(df_aligned)
    return scaled_array
