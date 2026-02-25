"""
Evaluation module — metrics, SHAP values, reports.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from pathlib import Path
from loguru import logger


# ── Regression metrics ────────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mse":  float(mean_squared_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ── Classification metrics ────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "f1":        float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall":    float(recall_score(y_true, y_pred)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 500) -> np.ndarray:
    """Compute SHAP values using TreeExplainer (works for LightGBM/XGBoost/CatBoost)."""
    X_sample = X.iloc[:max_samples] if len(X) > max_samples else X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # For classifiers, shap_values can be a list — take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values


def shap_top_features(model, X_row: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Return top-N SHAP contributors for a single prediction row.
    Used by the API for per-prediction explanation.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv.flatten()

    feature_names = list(X_row.columns)
    contributions = sorted(
        zip(feature_names, sv),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [
        {"feature": f, "shap_value": float(v), "direction": "+" if v > 0 else "-"}
        for f, v in contributions[:top_n]
    ]


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_shap_summary(shap_values, X: pd.DataFrame, output_path: str) -> str:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved → {output_path}")
    return output_path


def plot_predictions_vs_actual(y_true, y_pred, output_path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1)
    axes[0].set_xlabel("Actual RiskScore")
    axes[0].set_ylabel("Predicted RiskScore")
    axes[0].set_title("Predicted vs Actual")

    residuals = np.array(y_pred) - np.array(y_true)
    axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Predictions plot saved → {output_path}")
    return output_path


def plot_confusion_matrix(y_true, y_pred, output_path: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ── Report ───────────────────────────────────────────────────────────────────

def full_evaluation_report(
    reg_model,
    X_test: pd.DataFrame,
    y_test_reg,
    output_dir: str = "reports/",
) -> dict:
    """Run regression evaluation and generate plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Regression ──
    reg_pred = reg_model.predict(X_test)
    reg_metrics_ = regression_metrics(y_test_reg, reg_pred)
    logger.info(f"Regression  RMSE={reg_metrics_['rmse']:.4f}  R2={reg_metrics_['r2']:.4f}")

    plot_predictions_vs_actual(
        y_test_reg, reg_pred,
        str(output_dir / "regression_predictions.png")
    )

    # ── SHAP ──
    try:
        shap_vals = compute_shap_values(reg_model, X_test)
        plot_shap_summary(shap_vals, X_test.iloc[:500], str(output_dir / "shap_summary.png"))
        shap_plot = str(output_dir / "shap_summary.png")
    except Exception as e:
        logger.warning(f"SHAP plot skipped: {e}")
        shap_plot = None

    return {
        "regression": reg_metrics_,
        "plots": {
            "regression_pred": str(output_dir / "regression_predictions.png"),
            "shap_summary": shap_plot,
        }
    }


def full_evaluation_report_clf(
    clf_model,
    X_test: pd.DataFrame,
    y_test_clf,
    output_dir: str = "reports/",
) -> dict:
    """Run classification evaluation and generate plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clf_pred  = (clf_model.predict(X_test) >= 0.5).astype(int)
    clf_proba = clf_model.predict(X_test)
    clf_metrics_ = classification_metrics(y_test_clf, clf_pred, clf_proba)
    logger.info(f"Classification  AUC={clf_metrics_.get('roc_auc', 0):.4f}  F1={clf_metrics_['f1']:.4f}")

    plot_confusion_matrix(
        y_test_clf, clf_pred,
        str(output_dir / "confusion_matrix.png")
    )

    return {
        "classification": clf_metrics_,
        "plots": {
            "confusion_matrix": str(output_dir / "confusion_matrix.png"),
        }
    }