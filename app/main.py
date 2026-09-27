import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.model import ml_container
from app.routes import health, prediction, analytics

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management:
      1. Create SQLite database tables if not existing.
      2. Load and validate serialized ML artifacts (loan_model.pkl, scaler.pkl, feature_columns.pkl).
    """
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    logger.info("Loading and verifying ML model artifacts...")
    try:
        ml_container.load_artifacts()
    except Exception as e:
        logger.critical(f"FATAL: Failed to load ML artifacts: {e}")
        raise e

    logger.info("FastAPI application startup complete and ready for inference.")
    yield
    logger.info("FastAPI application shutting down.")

# Initialize FastAPI instance
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="Production-grade REST API for Loan Default Risk Prediction using Balanced Logistic Regression.",
    lifespan=lifespan
)

# Configure CORS: Allow local Vite dev server on any port (5173, 5174, etc.) and configurable FRONTEND_URL
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

if settings.FRONTEND_URL and settings.FRONTEND_URL not in allowed_origins:
    allowed_origins.append(settings.FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(analytics.router)
