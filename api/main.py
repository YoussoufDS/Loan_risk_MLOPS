"""
FastAPI application — Loan Risk & Approval prediction API.
"""

import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from loguru import logger

from api.schemas import (
    LoanApplicant,
    RiskPredictionResponse,
    ApprovalPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from api.predict import predict_risk, predict_approval, predict_batch, reload_models, get_models
from src.utils import load_config

app = FastAPI(
    title="Loan Risk MLOps API",
    description="Predict RiskScore (regression) and LoanApproved (classification) with SHAP explanations.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """API health check + model versions."""
    cfg = load_config()
    try:
        models, _ = get_models()
        return {
            "status": "ok",
            "regression_model_version": models.get("reg_version", "not loaded"),
            "classification_model_version": models.get("clf_version", "not loaded"),
            "mlflow_uri": cfg["mlflow"]["tracking_uri"],
        }
    except Exception as e:
        return {"status": f"degraded: {e}", "regression_model_version": None,
                "classification_model_version": None, "mlflow_uri": cfg["mlflow"]["tracking_uri"]}


# ── Predictions ───────────────────────────────────────────────────────────────

@app.post("/predict/risk", response_model=RiskPredictionResponse, tags=["Predictions"])
def predict_risk_endpoint(applicant: LoanApplicant):
    """
    Predict RiskScore for a single applicant.
    Returns score, risk level (Low/Medium/High), and SHAP top-5 features.
    """
    try:
        result = predict_risk(applicant.model_dump())
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Risk prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/approval", response_model=ApprovalPredictionResponse, tags=["Predictions"])
def predict_approval_endpoint(applicant: LoanApplicant):
    """
    Predict loan approval for a single applicant.
    Returns approval decision, probability, and SHAP top-5 features.
    """
    try:
        result = predict_approval(applicant.model_dump())
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Approval prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch_endpoint(file: UploadFile = File(...)):
    """
    Batch prediction from uploaded CSV file.
    Returns the enriched CSV with PredictedRiskScore, RiskLevel,
    ApprovalProbability, PredictedApproval columns added.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        result_df = predict_batch(df)

        # Stream result as CSV
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"},
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# ── Model Info ────────────────────────────────────────────────────────────────

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Return metadata of the active models (version, source, status)."""
    try:
        models, _ = get_models()
        return {
            "regression": {
                "version": models.get("reg_version"),
                "status": "loaded" if models.get("regression") else "unavailable",
            },
            "classification": {
                "version": models.get("clf_version"),
                "status": "loaded" if models.get("classification") else "unavailable",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload", tags=["Model"])
def reload_model_endpoint():
    """Force reload models from MLflow Registry (call after deploying a new version)."""
    reload_models()
    return {"message": "Model cache cleared. Models will reload on next request."}


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(
        "api.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=cfg["api"]["reload"],
    )