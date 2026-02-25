"""
Prediction logic — load models from MLflow Registry or local joblib fallback.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.preprocessing import feature_engineering, apply_encoders, apply_scaler
from src.evaluate import shap_top_features
from src.utils import load_config, load_artifact, ROOT


# ── Model cache (avoid reloading on every request) ───────────────────────────
_model_cache: dict = {}
_artifact_cache: dict = {}


def get_cfg():
    return load_config()


def load_model_from_registry(registry_name: str, stage: str = "Production"):
    """Load model from MLflow Model Registry."""
    cfg = get_cfg()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(registry_name, stages=[stage])
        if not versions:
            raise ValueError(f"No model in stage '{stage}' for '{registry_name}'")
        model_uri = f"models:/{registry_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        version = versions[0].version
        logger.info(f"Loaded '{registry_name}' v{version} from MLflow Registry ({stage})")
        return model, version
    except Exception as e:
        logger.warning(f"MLflow Registry unavailable ({e}). Falling back to local joblib.")
        return None, None


def load_preprocessing_artifacts() -> dict:
    """Load encoders + scaler from disk."""
    artifact_dir = ROOT / "models" / "artifacts"
    return {
        "encoders": load_artifact(artifact_dir / "encoders.joblib"),
        "scaler":   load_artifact(artifact_dir / "scaler.joblib"),
        "num_cols": load_artifact(artifact_dir / "num_cols.joblib"),
    }


def get_models() -> dict:
    """Return cached models, loading them if not yet cached."""
    global _model_cache, _artifact_cache
    cfg = get_cfg()

    if not _model_cache:
        logger.info("Loading models into cache...")

        # Try MLflow Registry first
        reg_model, reg_ver = load_model_from_registry(cfg["mlflow"]["model_registry_regression"])
        clf_model, clf_ver = load_model_from_registry(cfg["mlflow"]["model_registry_classification"])

        # Fallback to local joblib
        if reg_model is None:
            reg_path = ROOT / "models" / "regression_ensemble.joblib"
            if reg_path.exists():
                reg_model = load_artifact(reg_path)
                reg_ver = "local"
                logger.info("Regression model loaded from local joblib")

        if clf_model is None:
            clf_path = ROOT / "models" / "classification_ensemble.joblib"
            if clf_path.exists():
                clf_model = load_artifact(clf_path)
                clf_ver = "local"
                logger.info("Classification model loaded from local joblib")

        _model_cache = {
            "regression": reg_model,
            "classification": clf_model,
            "reg_version": reg_ver or "unknown",
            "clf_version": clf_ver or "unknown",
        }

    if not _artifact_cache:
        try:
            _artifact_cache = load_preprocessing_artifacts()
        except Exception as e:
            logger.error(f"Failed to load preprocessing artifacts: {e}")

    return _model_cache, _artifact_cache


def preprocess_single(applicant_dict: dict) -> pd.DataFrame:
    """
    Preprocess a single applicant dict:
    feature engineering → encoding → scaling.
    """
    _, artifacts = get_models()
    cfg = get_cfg()

    df = pd.DataFrame([applicant_dict])
    df = feature_engineering(df)
    df = apply_encoders(df, artifacts["encoders"], cfg)
    df = apply_scaler(df, artifacts["scaler"], artifacts["num_cols"])

    # Drop target columns if accidentally present
    targets = [cfg["features"]["target_regression"], cfg["features"]["target_classification"]]
    df = df.drop(columns=[c for c in targets if c in df.columns], errors="ignore")

    return df


def predict_risk(applicant_dict: dict) -> dict:
    """Predict RiskScore with SHAP explanation."""
    models, _ = get_models()
    reg_model = models["regression"]
    if reg_model is None:
        raise RuntimeError("Regression model not available. Run training first.")

    X = preprocess_single(applicant_dict)
    score = float(reg_model.predict(X)[0])

    # Risk level thresholds (adjust based on actual score distribution)
    if score < 30:
        risk_level = "Low"
    elif score < 60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # SHAP explanation (best individual model for SHAP)
    try:
        from src.utils import load_artifact, ROOT
        reg_models = load_artifact(ROOT / "models" / "regression_models.joblib")
        top_features = shap_top_features(reg_models["lgb"], X, top_n=5)
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        top_features = []

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "model_version": models["reg_version"],
        "shap_top_features": top_features,
    }


def predict_approval(applicant_dict: dict) -> dict:
    """Predict LoanApproved with probability and SHAP explanation."""
    models, _ = get_models()
    clf_model = models["classification"]
    if clf_model is None:
        raise RuntimeError("Classification model not available. Run training first.")

    X = preprocess_single(applicant_dict)
    proba = float(clf_model.predict(X)[0])   # WeightedEnsemble.predict returns probability
    approved = proba >= 0.5

    try:
        from src.utils import load_artifact, ROOT
        clf_models = load_artifact(ROOT / "models" / "classification_models.joblib")
        top_features = shap_top_features(clf_models["lgb"], X, top_n=5)
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        top_features = []

    return {
        "loan_approved": bool(approved),
        "approval_probability": proba,
        "model_version": models["clf_version"],
        "shap_top_features": top_features,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Batch prediction on a DataFrame. Returns df with added prediction columns."""
    models, _ = get_models()
    cfg = get_cfg()
    _, artifacts = get_models()

    df_proc = feature_engineering(df.copy())
    df_proc = apply_encoders(df_proc, artifacts["encoders"], cfg)
    df_proc = apply_scaler(df_proc, artifacts["scaler"], artifacts["num_cols"])

    targets = [cfg["features"]["target_regression"], cfg["features"]["target_classification"]]
    X = df_proc.drop(columns=[c for c in targets if c in df_proc.columns], errors="ignore")

    if models["regression"]:
        df["PredictedRiskScore"] = models["regression"].predict(X)
        df["RiskLevel"] = pd.cut(
            df["PredictedRiskScore"],
            bins=[-np.inf, 30, 60, np.inf],
            labels=["Low", "Medium", "High"]
        )

    if models["classification"]:
        proba = models["classification"].predict(X)
        df["ApprovalProbability"] = proba
        df["PredictedApproval"] = (proba >= 0.5).astype(int)

    return df


def reload_models():
    """Force reload models from registry (used after a new deployment)."""
    global _model_cache, _artifact_cache
    _model_cache = {}
    _artifact_cache = {}
    logger.info("Model cache cleared — will reload on next request")