import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base backend directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application configuration settings loaded from environment or .env file."""
    
    APP_TITLE: str = "Loan Default Prediction API"
    APP_VERSION: str = "1.0.0"
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'loan_predictions.db'}"
    
    # CORS
    FRONTEND_URL: str = "http://localhost:5173"
    
    # ML Artifacts Paths
    MODEL_PATH: str = str(BASE_DIR / "ml_models" / "loan_model.pkl")
    SCALER_PATH: str = str(BASE_DIR / "ml_models" / "scaler.pkl")
    FEATURE_COLUMNS_PATH: str = str(BASE_DIR / "ml_models" / "feature_columns.pkl")
    
    # Model Metadata
    MODEL_NAME: str = "Balanced Logistic Regression"
    MODEL_TYPE: str = "Logistic Regression"
    CLASS_WEIGHT: str = "balanced"
    EXPECTED_FEATURES_COUNT: int = 24

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
