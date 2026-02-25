"""
Hyperparameter optimization (Optuna) + Ensemble weight optimization (Hill Climbing).

Key invariants:
  - Optuna uses Val-B ONLY  →  never touches Val-A, Val-C, or Test
  - Hill Climbing uses Val-C ONLY  →  never touches Val-A, Val-B, or Test
  - Early stopping uses Val-A ONLY  →  never used for selection decisions
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.metrics import mean_squared_error, roc_auc_score
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from loguru import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA — REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def build_optuna_objective_regression(X_train, y_train, X_val_a, y_val_a, X_val_b, y_val_b, cfg):
    """
    Returns an Optuna objective that:
      - trains on X_train with early stopping on Val-A
      - evaluates on Val-B (the objective score)
    """
    early_rounds = cfg["training"]["early_stopping_rounds"]

    def objective(trial):
        # ── LightGBM ──
        lgb_params = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 500, 5000, step=500),
            "learning_rate": trial.suggest_float("lgb_lr", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("lgb_depth", 3, 10),
            "num_leaves": trial.suggest_int("lgb_leaves", 20, 300),
            "min_child_samples": trial.suggest_int("lgb_min_child", 5, 100),
            "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("lgb_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lgb_lambda", 1e-4, 10.0, log=True),
            "random_state": 42, "verbose": -1,
            "callbacks": [optuna.integration.lightgbm.LightGBMPruningCallback(trial, "rmse")],
        }
        lgb = LGBMRegressor(**{k: v for k, v in lgb_params.items() if k != "callbacks"})
        lgb.set_params(callbacks=lgb_params["callbacks"])
        lgb.fit(
            X_train, y_train,
            eval_set=[(X_val_a, y_val_a)],
            eval_metric="rmse",
            callbacks=[
                optuna.integration.lightgbm.LightGBMPruningCallback(trial, "rmse"),
                __import__("lightgbm").early_stopping(early_rounds, verbose=False),
                __import__("lightgbm").log_evaluation(-1),
            ],
        )
        pred = lgb.predict(X_val_b)
        return float(np.sqrt(mean_squared_error(y_val_b, pred)))

    return objective


def build_optuna_objective_classification(X_train, y_train, X_val_a, y_val_a, X_val_b, y_val_b, cfg):
    """Optuna objective for binary classification — optimizes ROC-AUC on Val-B."""
    early_rounds = cfg["training"]["early_stopping_rounds"]

    def objective(trial):
        lgb_params = {
            "n_estimators": trial.suggest_int("lgb_n_est", 500, 5000, step=500),
            "learning_rate": trial.suggest_float("lgb_lr", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("lgb_depth", 3, 10),
            "num_leaves": trial.suggest_int("lgb_leaves", 20, 300),
            "min_child_samples": trial.suggest_int("lgb_min_child", 5, 100),
            "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("lgb_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lgb_lambda", 1e-4, 10.0, log=True),
            "random_state": 42, "verbose": -1,
        }
        lgb = LGBMClassifier(**lgb_params)
        lgb.fit(
            X_train, y_train,
            eval_set=[(X_val_a, y_val_a)],
            eval_metric="auc",
            callbacks=[
                __import__("lightgbm").early_stopping(early_rounds, verbose=False),
                __import__("lightgbm").log_evaluation(-1),
            ],
        )
        proba = lgb.predict_proba(X_val_b)[:, 1]
        # Return negative AUC (Optuna minimizes)
        return -float(roc_auc_score(y_val_b, proba))

    return objective


def run_optuna(objective_fn, cfg, study_name: str) -> optuna.Study:
    """Run Optuna study with TPE sampler + Hyperband pruner."""
    storage = cfg["training"]["optuna_db"]
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3),
        load_if_exists=True,
    )
    study.optimize(
        objective_fn,
        n_trials=cfg["training"]["optuna_trials"],
        timeout=cfg["training"]["optuna_timeout"],
        show_progress_bar=True,
        gc_after_trial=True,
    )
    logger.info(f"Optuna [{study_name}] best value: {study.best_value:.5f}")
    logger.info(f"Best params: {study.best_params}")
    return study


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN FINAL MODELS with best params
# ═══════════════════════════════════════════════════════════════════════════════

def train_final_regressors(best_params: dict, X_train, y_train, X_val_a, y_val_a, cfg) -> dict:
    """
    Train LightGBM, XGBoost, CatBoost regressors with best Optuna params.
    Early stopping uses Val-A only.
    """
    er = cfg["training"]["early_stopping_rounds"]
    models = {}

    # ── LightGBM ──
    logger.info("Training LightGBM regressor...")
    lgb_p = {k.replace("lgb_", ""): v for k, v in best_params.items() if k.startswith("lgb_")}
    lgb_p.update({"random_state": 42, "verbose": -1})
    lgb = LGBMRegressor(**lgb_p)
    lgb.fit(
        X_train, y_train,
        eval_set=[(X_val_a, y_val_a)],
        eval_metric="rmse",
        callbacks=[
            __import__("lightgbm").early_stopping(er, verbose=False),
            __import__("lightgbm").log_evaluation(-1),
        ],
    )
    models["lgb"] = lgb

    # ── XGBoost ──
    logger.info("Training XGBoost regressor...")
    xgb = XGBRegressor(
        n_estimators=5000, learning_rate=0.01, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.01, reg_lambda=9.0, eval_metric="rmse",
        random_state=42, verbosity=0, early_stopping_rounds=er,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val_a, y_val_a)], verbose=False)
    models["xgb"] = xgb

    # ── CatBoost ──
    logger.info("Training CatBoost regressor...")
    cat = CatBoostRegressor(
        iterations=4000, learning_rate=0.017, depth=6,
        l2_leaf_reg=0.09, bagging_temperature=0.4, border_count=221,
        random_strength=6.6, eval_metric="RMSE", random_seed=42,
        verbose=0, early_stopping_rounds=er,
    )
    cat.fit(X_train, y_train, eval_set=(X_val_a, y_val_a))
    models["cat"] = cat

    return models


def train_final_classifiers(best_params: dict, X_train, y_train, X_val_a, y_val_a, cfg) -> dict:
    """Train LightGBM, XGBoost, CatBoost classifiers."""
    er = cfg["training"]["early_stopping_rounds"]
    models = {}

    logger.info("Training LightGBM classifier...")
    lgb_p = {k.replace("lgb_", ""): v for k, v in best_params.items() if k.startswith("lgb_")}
    lgb_p.update({"random_state": 42, "verbose": -1})
    lgb = LGBMClassifier(**lgb_p)
    lgb.fit(
        X_train, y_train,
        eval_set=[(X_val_a, y_val_a)],
        eval_metric="auc",
        callbacks=[
            __import__("lightgbm").early_stopping(er, verbose=False),
            __import__("lightgbm").log_evaluation(-1),
        ],
    )
    models["lgb"] = lgb

    logger.info("Training XGBoost classifier...")
    xgb = XGBClassifier(
        n_estimators=3000, learning_rate=0.01, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        use_label_encoder=False, random_state=42, verbosity=0,
        early_stopping_rounds=er,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val_a, y_val_a)], verbose=False)
    models["xgb"] = xgb

    logger.info("Training CatBoost classifier...")
    cat = CatBoostClassifier(
        iterations=3000, learning_rate=0.02, depth=6,
        eval_metric="AUC", random_seed=42,
        verbose=0, early_stopping_rounds=er,
    )
    cat.fit(X_train, y_train, eval_set=(X_val_a, y_val_a))
    models["cat"] = cat

    return models


# ═══════════════════════════════════════════════════════════════════════════════
# HILL CLIMBING — pure numpy (no GPU dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _rmse_batch(actual: np.ndarray, predicted_matrix: np.ndarray) -> np.ndarray:
    """Vectorized RMSE over multiple prediction columns."""
    if actual.ndim == 1:
        actual = actual[:, np.newaxis]
    return np.sqrt(np.mean((actual - predicted_matrix) ** 2, axis=0))


def hill_climbing_weights(
    models: dict,
    X_val_c,
    y_val_c: np.ndarray,
    cfg: dict,
    task: str = "regression",
) -> dict:
    """
    Optimize ensemble weights on Val-C using Hill Climbing.
    Returns dict: {model_name: weight}
    """
    model_names = list(models.keys())
    use_neg = bool(cfg["training"]["hill_climb_negative_weights"])
    max_models = int(cfg["training"]["hill_climb_max_models"])
    tol = float(cfg["training"]["hill_climb_tol"])

    # ── Build predictions matrix ──
    preds_matrix = np.stack([
        models[n].predict(X_val_c) if task == "regression"
        else models[n].predict_proba(X_val_c)[:, 1]
        for n in model_names
    ]).T  # shape: (n_samples, n_models)

    y_arr = np.array(y_val_c)

    # ── Score each individual model ──
    best_score = np.inf
    best_idx = 0
    metric_fn = _rmse if task == "regression" else lambda a, p: -roc_auc_score(a, p)

    for k, name in enumerate(model_names):
        score = metric_fn(y_arr, preds_matrix[:, k])
        logger.info(f"  Single model [{name}]: score = {score:.5f}")
        if score < best_score:
            best_score = score
            best_idx = k

    logger.info(f"Best single model: {model_names[best_idx]} (score={best_score:.5f})")

    # ── Hill Climbing ──
    start = -0.50 if use_neg else 0.01
    ww = np.arange(start, 0.51, 0.01)

    best_ensemble = preds_matrix[:, best_idx].copy()
    chosen_models = [best_idx]
    chosen_weights = []
    old_best_score = best_score

    for iteration in range(1_000_000):
        step_best_score = best_score
        step_best_idx = -1
        step_best_w = 0.0
        step_best_ensemble = None

        for k in range(len(model_names)):
            new_pred = preds_matrix[:, k]
            # Test all weight candidates at once (vectorized)
            m1 = np.outer(best_ensemble, np.ones(len(ww))) * (1 - ww)
            m2 = np.outer(new_pred, np.ones(len(ww))) * ww
            mm = m1 + m2

            scores = _rmse_batch(y_arr, mm)
            best_w_idx = int(np.argmin(scores))
            score_candidate = float(scores[best_w_idx])

            if score_candidate < step_best_score:
                step_best_score = score_candidate
                step_best_idx = k
                step_best_w = float(ww[best_w_idx])
                step_best_ensemble = mm[:, best_w_idx]

        if step_best_idx == -1 or (best_score - step_best_score) < tol:
            logger.info(f"Hill Climbing converged at iteration {iteration} (tol={tol})")
            break

        best_score = step_best_score
        best_ensemble = step_best_ensemble
        chosen_models.append(step_best_idx)
        chosen_weights.append(step_best_w)
        logger.info(
            f"  Iter {iteration+1}: +{model_names[step_best_idx]} "
            f"w={step_best_w:.3f} → score={best_score:.5f}"
        )

        if len(chosen_models) >= max_models:
            logger.info(f"Reached max_models={max_models}")
            break

    # ── Compute final weights per model ──
    raw_weights = np.array([1.0])
    for w in chosen_weights:
        raw_weights = raw_weights * (1 - w)
        raw_weights = np.append(raw_weights, w)

    # Aggregate by model name
    weight_map = {n: 0.0 for n in model_names}
    for m_idx, w in zip(chosen_models, raw_weights):
        weight_map[model_names[m_idx]] += w

    logger.info(f"Final ensemble weights: {weight_map}")
    logger.info(f"Final ensemble score on Val-C: {best_score:.5f}")
    return weight_map