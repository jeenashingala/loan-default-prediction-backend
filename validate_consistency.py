"""
ML Consistency Validation:
Compares direct notebook-style preprocessing & model execution
versus FastAPI backend preprocessing & model execution.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import joblib

from app.schemas import LoanPredictionRequest
from app.preprocessing import preprocess_features
from app.model import ml_container

def run_consistency_check():
    print("=== STARTING ML CONSISTENCY VALIDATION ===")
    
    # 1. Load artifacts directly (Notebook style)
    model_direct = joblib.load(BASE_DIR / "ml_models" / "loan_model.pkl")
    scaler_direct = joblib.load(BASE_DIR / "ml_models" / "scaler.pkl")
    cols_direct = joblib.load(BASE_DIR / "ml_models" / "feature_columns.pkl")
    
    sample_raw = {
        "Age": 38.0,
        "Income": 65000.0,
        "LoanAmount": 28000.0,
        "CreditScore": 680.0,
        "MonthsEmployed": 45.0,
        "NumCreditLines": 4.0,
        "InterestRate": 11.5,
        "LoanTerm": 36.0,
        "DTIRatio": 0.38,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Home",
        "HasCoSigner": "No"
    }

    # Method A: Direct Notebook Preprocessing
    df_nb = pd.DataFrame([sample_raw])
    cat_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose", "HasMortgage", "HasDependents", "HasCoSigner"]
    df_nb_encoded = pd.get_dummies(df_nb, columns=cat_cols, dtype=int)
    df_nb_aligned = df_nb_encoded.reindex(columns=cols_direct, fill_value=0)
    scaled_nb = scaler_direct.transform(df_nb_aligned)
    pred_nb = model_direct.predict(scaled_nb)
    proba_nb = model_direct.predict_proba(scaled_nb)

    # Method B: FastAPI Preprocessing & Container
    ml_container.load_artifacts()
    req = LoanPredictionRequest(**sample_raw)
    scaled_api = preprocess_features(req, ml_container.get_feature_columns(), ml_container.get_scaler())
    pred_api = ml_container.get_model().predict(scaled_api)
    proba_api = ml_container.get_model().predict_proba(scaled_api)

    # Assertions
    print("\nComparing Notebook vs FastAPI Preprocessing:")
    print("Feature columns identical?:", cols_direct == ml_container.get_feature_columns())
    assert cols_direct == ml_container.get_feature_columns(), "Feature column list mismatch!"
    
    print("Scaled array shapes match?:", scaled_nb.shape == scaled_api.shape)
    assert scaled_nb.shape == scaled_api.shape, "Shape mismatch!"

    max_diff = np.max(np.abs(scaled_nb - scaled_api))
    print(f"Max absolute difference in scaled features: {max_diff:.10e}")
    assert max_diff < 1e-12, f"Scaled values differ by {max_diff}!"

    print("Predictions match?:", pred_nb[0] == pred_api[0])
    assert pred_nb[0] == pred_api[0], "Prediction mismatch!"

    proba_diff = np.max(np.abs(proba_nb - proba_api))
    print(f"Max absolute difference in predict_proba: {proba_diff:.10e}")
    assert proba_diff < 1e-12, f"Probabilities differ by {proba_diff}!"

    print(f"Prediction: {pred_api[0]} (No Default = 0, Default = 1)")
    print(f"Default Probability: {proba_api[0][1]:.4f}")
    print(f"No-Default Probability: {proba_api[0][0]:.4f}")
    print("\n=== ML CONSISTENCY VALIDATION 100% PERFECT! ===")

if __name__ == "__main__":
    run_consistency_check()
