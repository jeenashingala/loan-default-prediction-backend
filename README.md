# Loan Default Prediction — FastAPI Backend

Production-ready REST API microservice built with **FastAPI**, **SQLAlchemy**, and **Scikit-Learn (1.6.1)** powering real-time credit default risk predictions and historical audit analytics.

---

## 1. Architecture Overview

- **Framework**: FastAPI 0.110+
- **ASGI Server**: Uvicorn
- **Persistence**: SQLite (Local) / PostgreSQL-ready through `DATABASE_URL` via SQLAlchemy
- **Data Validation**: Pydantic v2
- **Model Engine**: Balanced Logistic Regression (`scikit-learn 1.6.1`)
- **Preprocessing**: 24-feature pipeline aligned with StandardScaler normalization

---

## 2. Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # Application entrypoint & CORS middleware
│   ├── config.py            # Environment settings (Pydantic BaseSettings)
│   ├── database.py          # SQLAlchemy connection & session manager
│   ├── model.py             # In-memory ML singleton & artifact validation
│   ├── preprocessing.py     # Feature engineering, dummy alignment, scaling
│   ├── schemas.py           # Pydantic request/response schemas
│   ├── models/
│   │   ├── __init__.py
│   │   └── prediction.py    # PredictionRecord SQLAlchemy DB model
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py        # / and /health endpoints
│   │   ├── prediction.py    # /predict and /predictions CRUD endpoints
│   │   └── analytics.py     # /analytics/dashboard, /analytics/metrics, /model
│   └── services/
│       ├── __init__.py
│       ├── prediction_service.py # Core ML scoring & persistence logic
│       └── analytics_service.py  # Dashboard aggregates & cohort stats
├── ml_models/
│   ├── loan_model.pkl       # Trained Balanced Logistic Regression model
│   ├── scaler.pkl           # Trained StandardScaler (24 features)
│   └── feature_columns.pkl  # Exact 24 feature order source of truth
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 3. Local Setup & Running Instructions

### Prerequisites
- Python 3.10+ (Python 3.13 recommended)

### Step 1: Create and Activate Virtual Environment
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### Step 4: Run FastAPI Server
```bash
uvicorn app.main:app --reload
```

Server will start on **`http://localhost:8000`**.
Interactive Swagger API documentation is accessible at **`http://localhost:8000/docs`**.

---

## 4. API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root heartbeat |
| `GET` | `/health` | API and model status (`healthy`, `model_loaded`) |
| `POST` | `/predict` | Ingests 16 applicant attributes, runs inference, returns calibrated probability and stores record |
| `GET` | `/predictions` | Returns paginated prediction records with search, filter, and sorting |
| `GET` | `/predictions/{id}` | Fetches individual applicant prediction record |
| `DELETE` | `/predictions/{id}` | Removes a prediction record |
| `GET` | `/analytics/dashboard` | Computes live dashboard metrics (counts, default rate, 7-day trend) |
| `GET` | `/analytics/metrics` | Computes segmentation across employment type, purpose, and scatter points |
| `GET` | `/model` | Returns active ML model metadata, scaler type, and 24 features |

---

## 5. Deployment Guide (Render)

1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. **Environment Variables**:
   - `DATABASE_URL`: `sqlite:///./loan_predictions.db` (or managed PostgreSQL connection string)
   - `FRONTEND_URL`: URL of your deployed frontend (e.g. `https://your-app.vercel.app`)
   - `MODEL_PATH`: `ml_models/loan_model.pkl`
   - `SCALER_PATH`: `ml_models/scaler.pkl`
   - `FEATURE_COLUMNS_PATH`: `ml_models/feature_columns.pkl`
