import logging
from pathlib import Path
from typing import List, Tuple, Any
import joblib
from app.config import settings

logger = logging.getLogger(__name__)

class ModelContainer:
    """Singleton container managing trained ML artifacts in memory."""
    
    def __init__(self):
        self.model: Any = None
        self.scaler: Any = None
        self.feature_columns: List[str] = []
        self._loaded: bool = False

    def load_artifacts(self) -> None:
        """Load and strictly validate the three serialized ML artifacts."""
        model_path = Path(settings.MODEL_PATH)
        scaler_path = Path(settings.SCALER_PATH)
        cols_path = Path(settings.FEATURE_COLUMNS_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}")
        if not cols_path.exists():
            raise FileNotFoundError(f"Feature columns artifact not found at {cols_path}")

        logger.info("Loading ML artifacts from disk...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(cols_path)

        # Strict validation as defined in requirements
        expected_count = settings.EXPECTED_FEATURES_COUNT
        cols_len = len(self.feature_columns)
        scaler_feats = getattr(self.scaler, "n_features_in_", None)
        model_feats = getattr(self.model, "n_features_in_", None)

        if cols_len != expected_count:
            raise ValueError(
                f"Feature columns count mismatch: expected {expected_count}, got {cols_len}"
            )

        if scaler_feats != expected_count:
            raise ValueError(
                f"Scaler n_features_in_ mismatch: expected {expected_count}, got {scaler_feats}"
            )

        if model_feats != expected_count:
            raise ValueError(
                f"Model n_features_in_ mismatch: expected {expected_count}, got {model_feats}"
            )

        self._loaded = True
        logger.info(
            f"Successfully loaded and verified ML artifacts: 24 features aligned with {settings.MODEL_NAME}."
        )

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model(self) -> Any:
        if not self._loaded:
            raise RuntimeError("Model has not been loaded yet.")
        return self.model

    def get_scaler(self) -> Any:
        if not self._loaded:
            raise RuntimeError("Scaler has not been loaded yet.")
        return self.scaler

    def get_feature_columns(self) -> List[str]:
        if not self._loaded:
            raise RuntimeError("Feature columns have not been loaded yet.")
        return self.feature_columns

# Global container instance
ml_container = ModelContainer()
