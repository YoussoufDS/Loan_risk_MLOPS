"""
Main training orchestrator.
  1. Preprocessing (nested splits)
  2. Optuna optimization (Val-B)
  3. Train final models (early stop on Val-A)
  4. Hill Climbing weights (Val-C)
  5. Evaluate on Test
  6. Log everything to MLflow + register model
"""

import argparse
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from pathlib import Path
from loguru import logger

from src.preprocessing import run_preprocessing
from src.optimize import (
    build_optuna_objective_regression,
    build_optuna_objective_classification,
    run_optuna,
    train_final_regressors,
    train_final_classifiers,
    hill_climbing_weights,
)
from src.evaluate import full_evaluation_report, full_evaluation_report_clf, regression_metrics, classification_metrics
from src.utils import load_config, save_artifact, ROOT


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICT WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class WeightedEnsemble:
    """Thin wrapper to combine model predictions with Hill Climbing weights."""

    def __init__(self, models: dict, weights: dict, task: str = "regression"):
        self.models = models
        self.weights = weights
        self.task = task

    def predict(self, X):
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if self.task == "regression":
                preds += w * model.predict(X)
            else:
                preds += w * model.predict_proba(X)[:, 1]
        return preds

    def predict_proba(self, X):
        proba = self.predict(X)
        return np.column_stack([1 - proba, proba])

    def predict_class(self, X, threshold: float = 0.5):
        return (self.predict(X) >= threshold).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# MLflow helpers
# ═══════════════════════════════════════════════════════════════════════════════

def setup_mlflow(cfg: dict) -> None:
    import os
    # Priorité : variable d'environnement > config.yaml
    uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI: {uri}")

    # Force artifact root to avoid Windows absolute paths on Linux runners
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    if artifact_root:
        os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_root
        logger.info(f"MLflow artifact root: {artifact_root}")


def log_and_register(
    run,
    model,
    model_name: str,
    metrics: dict,
    params: dict,
    artifacts: dict,
    X_sample,
    cfg: dict,
) -> None:
    """Log params, metrics, artifacts and register model in MLflow Registry."""
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    for artifact_path in artifacts.values():
        if artifact_path is None:
            continue
        # Résoudre depuis le répertoire courant (pas depuis ROOT)
        # Cela évite d'hériter de chemins Windows absolus
        p = Path(artifact_path)
        if p.is_absolute():
            # Sur Linux runner, un chemin absolu Windows comme C:\... est invalide
            # On reconstruit depuis le cwd
            try:
                p = Path.cwd() / p.relative_to(ROOT)
            except (ValueError, Exception):
                p = Path.cwd() / "reports" / p.name
        else:
            p = Path.cwd() / p
        if p.exists():
            mlflow.log_artifact(str(p))
        else:
            logger.warning(f"Artifact not found, skipping: {p}")

    # Infer signature for schema validation at inference
    signature = infer_signature(X_sample, model.predict(X_sample))
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name=model_name,
    )
    logger.info(f"Model registered in MLflow Registry as '{model_name}'")


def auto_promote(cfg: dict, new_rmse: float, registry_name: str) -> None:
    """Promote to Production if new model beats current Production by threshold."""
    client = mlflow.MlflowClient()
    threshold = cfg["mlflow"]["auto_promote_threshold"]

    try:
        prod_versions = client.get_latest_versions(registry_name, stages=["Production"])
        if not prod_versions:
            # No production model yet — promote automatically
            versions = client.get_latest_versions(registry_name, stages=["None"])
            if versions:
                client.transition_model_version_stage(
                    name=registry_name, version=versions[-1].version, stage="Production"
                )
                logger.info(f"First model promoted to Production (v{versions[-1].version})")
            return

        prod_run_id = prod_versions[0].run_id
        prod_metrics = client.get_run(prod_run_id).data.metrics
        prod_rmse = prod_metrics.get("test_rmse", float("inf"))

        improvement = (prod_rmse - new_rmse) / prod_rmse
        logger.info(f"Production RMSE={prod_rmse:.5f} | New RMSE={new_rmse:.5f} | Δ={improvement:.3%}")

        if improvement > threshold:
            # Demote current prod to Archived
            client.transition_model_version_stage(
                name=registry_name, version=prod_versions[0].version, stage="Archived"
            )
            # Promote new to Production
            new_versions = client.get_latest_versions(registry_name, stages=["None"])
            if new_versions:
                client.transition_model_version_stage(
                    name=registry_name, version=new_versions[-1].version, stage="Production"
                )
                logger.info(f"New model promoted to Production (RMSE improved {improvement:.2%})")
        else:
            logger.warning(f"New model NOT promoted — improvement {improvement:.2%} < threshold {threshold:.0%}")

    except Exception as e:
        logger.error(f"Auto-promote failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

def train(raw_path: str = None, trigger: str = "manual") -> dict:
    cfg = load_config()
    setup_mlflow(cfg)

    # ── 1. Preprocessing ──
    logger.info("STEP 1/6 — Preprocessing")
    data = run_preprocessing(raw_path)

    X_train  = data["X_train"]
    X_val_a  = data["X_val_a"]
    X_val_b  = data["X_val_b"]
    X_val_c  = data["X_val_c"]
    X_test   = data["X_test"]

    y_train_reg  = data["y_train_reg"]
    y_val_a_reg  = data["y_val_a_reg"]
    y_val_b_reg  = data["y_val_b_reg"]
    y_val_c_reg  = data["y_val_c_reg"]
    y_test_reg   = data["y_test_reg"]

    y_train_clf  = data["y_train_clf"]
    y_val_a_clf  = data["y_val_a_clf"]
    y_val_b_clf  = data["y_val_b_clf"]
    y_val_c_clf  = data["y_val_c_clf"]
    y_test_clf   = data["y_test_clf"]

    results = {}

    # ════════════════════════════════════════════════════════
    # REGRESSION RUN
    # ════════════════════════════════════════════════════════
    mlflow.set_experiment(cfg["mlflow"]["experiment_regression"])

    with mlflow.start_run(run_name=f"regression_{trigger}") as run:
        mlflow.set_tags({"trigger": trigger, "task": "regression"})

        # ── 2. Optuna (Val-B) ──
        logger.info("STEP 2/6 — Optuna regression (Val-B)")
        obj_reg = build_optuna_objective_regression(
            X_train, y_train_reg, X_val_a, y_val_a_reg, X_val_b, y_val_b_reg, cfg
        )
        study_reg = run_optuna(obj_reg, cfg, study_name="regression")

        # ── 3. Train final models (Val-A for early stop) ──
        logger.info("STEP 3/6 — Train final regressors")
        reg_models = train_final_regressors(
            study_reg.best_params, X_train, y_train_reg, X_val_a, y_val_a_reg, cfg
        )

        # ── 4. Hill Climbing (Val-C) ──
        logger.info("STEP 4/6 — Hill Climbing weights (Val-C)")
        reg_weights = hill_climbing_weights(reg_models, X_val_c, y_val_c_reg, cfg, task="regression")

        # ── Ensemble ──
        reg_ensemble = WeightedEnsemble(reg_models, reg_weights, task="regression")

        # ── 5. Evaluate on Test ──
        logger.info("STEP 5/6 — Evaluate on Test set")
        test_pred_reg = reg_ensemble.predict(X_test)
        test_metrics_reg = regression_metrics(y_test_reg, test_pred_reg)
        logger.info(f"Test RMSE={test_metrics_reg['rmse']:.4f}  R2={test_metrics_reg['r2']:.4f}")

        # ── 6. MLflow logging ──
        logger.info("STEP 6/6 — MLflow logging & registration")

        # Log artifacts
        save_artifact(reg_ensemble, ROOT / "models" / "regression_ensemble.joblib")
        save_artifact(reg_models, ROOT / "models" / "regression_models.joblib")

        report = full_evaluation_report(
            reg_models["lgb"],
            X_test, y_test_reg,
            output_dir=str(ROOT / "reports")
        )

        params_to_log = {
            "optuna_best_value": study_reg.best_value,
            "n_optuna_trials": len(study_reg.trials),
            **{f"weight_{k}": v for k, v in reg_weights.items()},
            **{f"best_{k}": v for k, v in study_reg.best_params.items()},
        }
        metrics_to_log = {f"test_{k}": v for k, v in test_metrics_reg.items()}

        log_and_register(
            run=run,
            model=reg_ensemble,
            model_name=cfg["mlflow"]["model_registry_regression"],
            metrics=metrics_to_log,
            params=params_to_log,
            artifacts=report["plots"],
            X_sample=X_test.iloc[:10],
            cfg=cfg,
        )

        auto_promote(cfg, test_metrics_reg["rmse"], cfg["mlflow"]["model_registry_regression"])
        results["regression"] = {"metrics": test_metrics_reg, "run_id": run.info.run_id}

    # ════════════════════════════════════════════════════════
    # CLASSIFICATION RUN
    # ════════════════════════════════════════════════════════
    mlflow.set_experiment(cfg["mlflow"]["experiment_classification"])

    with mlflow.start_run(run_name=f"classification_{trigger}") as run:
        mlflow.set_tags({"trigger": trigger, "task": "classification"})

        obj_clf = build_optuna_objective_classification(
            X_train, y_train_clf, X_val_a, y_val_a_clf, X_val_b, y_val_b_clf, cfg
        )
        study_clf = run_optuna(obj_clf, cfg, study_name="classification")

        clf_models = train_final_classifiers(
            study_clf.best_params, X_train, y_train_clf, X_val_a, y_val_a_clf, cfg
        )
        clf_weights = hill_climbing_weights(clf_models, X_val_c, y_val_c_clf, cfg, task="classification")
        clf_ensemble = WeightedEnsemble(clf_models, clf_weights, task="classification")

        test_proba_clf = clf_ensemble.predict(X_test)
        test_pred_clf  = (test_proba_clf >= 0.5).astype(int)
        test_metrics_clf = classification_metrics(y_test_clf, test_pred_clf, test_proba_clf)
        logger.info(f"Test AUC={test_metrics_clf.get('roc_auc', 0):.4f}  F1={test_metrics_clf['f1']:.4f}")

        save_artifact(clf_ensemble, ROOT / "models" / "classification_ensemble.joblib")
        save_artifact(clf_models,   ROOT / "models" / "classification_models.joblib")

        clf_report = full_evaluation_report_clf(
            clf_ensemble, X_test, y_test_clf,
            output_dir=str(ROOT / "reports")
        )

        log_and_register(
            run=run,
            model=clf_ensemble,
            model_name=cfg["mlflow"]["model_registry_classification"],
            metrics={f"test_{k}": v for k, v in test_metrics_clf.items()},
            params={
                "optuna_best_value": study_clf.best_value,
                **{f"weight_{k}": v for k, v in clf_weights.items()},
            },
            artifacts=clf_report["plots"],
            X_sample=X_test.iloc[:10],
            cfg=cfg,
        )
        results["classification"] = {"metrics": test_metrics_clf, "run_id": run.info.run_id}

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Regression  RMSE : {results['regression']['metrics']['rmse']:.4f}")
    logger.info(f"  Classification AUC: {results['classification']['metrics'].get('roc_auc', 0):.4f}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to raw CSV")
    parser.add_argument("--trigger", type=str, default="manual")
    args = parser.parse_args()
    train(raw_path=args.data, trigger=args.trigger)