"""
Preprocessing pipeline — feature engineering, encoding, scaling.
All transformers are fit on Train only, then applied to all splits.
Artifacts saved for inference reproducibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from loguru import logger

from src.utils import load_config, save_artifact, ROOT


# ── Feature Engineering ───────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create financial ratio features. Safe division avoids div-by-zero."""
    df = df.copy()

    eps = 1e-9  # avoid division by zero

    df["AnIncomeToAssetsRatio"]       = df["AnnualIncome"] / (df["TotalAssets"] + eps)
    df["AnExperienceToAnIncomeRatio"] = df["Experience"]   / (df["AnnualIncome"] + eps)
    df["LoantoAnIncomeRatio"]         = df["LoanAmount"]   / (df["AnnualIncome"] + eps)
    df["DependetToAnIncomeRatio"]     = df["AnnualIncome"] / (df["NumberOfDependents"] + 1)
    df["LoansToAssetsRatio"]          = df["TotalLiabilities"] / (df["TotalAssets"] + eps)
    df["LoanPaymentToIncomeRatio"]    = df["MonthlyLoanPayment"] / (df["MonthlyIncome"] + eps)
    df["AnIncomeToDepts"]             = df["AnnualIncome"] / (
        df["MonthlyLoanPayment"] * 12 + df["MonthlyDebtPayments"] * 12 + eps
    )
    df["AssetsToLoan"]                = df["TotalAssets"] / (
        df["TotalLiabilities"] + df["LoanAmount"] + eps
    )

    # Temporal features (meaningful for real data; neutral on synthetic)
    if "ApplicationDate" in df.columns:
        df["AppMonth"]   = pd.to_datetime(df["ApplicationDate"]).dt.month
        df["AppQuarter"] = pd.to_datetime(df["ApplicationDate"]).dt.quarter
        df["AppDayOfWeek"] = pd.to_datetime(df["ApplicationDate"]).dt.dayofweek
        df.drop(columns=["ApplicationDate"], inplace=True)  # remove string col

    return df


# ── Split ─────────────────────────────────────────────────────────────────────

def nested_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """
    5-partition nested split that prevents validation contamination:
      Train 60% | Val-A 10% | Val-B 10% | Val-C 10% | Test 10%
    """
    s = cfg["data"]["split"]
    seed = cfg["data"]["random_state"]

    # Step 1: carve out Test
    rest, test = train_test_split(df, test_size=s["test"], random_state=seed)

    # Step 2: carve out Val-C (Hill Climbing)
    rest, val_c = train_test_split(rest, test_size=s["val_hill_climb"] / (1 - s["test"]), random_state=seed)

    # Step 3: carve out Val-B (Optuna)
    ratio_b = s["val_optuna"] / (1 - s["test"] - s["val_hill_climb"])
    rest, val_b = train_test_split(rest, test_size=ratio_b, random_state=seed)

    # Step 4: carve out Val-A (early stopping)
    ratio_a = s["val_early_stop"] / (1 - s["test"] - s["val_hill_climb"] - s["val_optuna"])
    train, val_a = train_test_split(rest, test_size=ratio_a, random_state=seed)

    splits = {"train": train, "val_a": val_a, "val_b": val_b, "val_c": val_c, "test": test}

    for name, split in splits.items():
        pct = len(split) / len(df) * 100
        logger.info(f"  Split {name:8s}: {len(split):5d} rows ({pct:.1f}%)")

    return splits


# ── Column Types ──────────────────────────────────────────────────────────────

def create_col_types(df: pd.DataFrame, cfg: dict) -> tuple[list, list]:
    """Separate categorical and numerical columns."""
    thr = cfg["features"]["numerical_threshold_cat"]
    targets = [cfg["features"]["target_regression"], cfg["features"]["target_classification"]]
    drop = cfg["features"]["drop_cols"] + targets

    cat_cols = [c for c in df.columns if df[c].dtype == "O" and c not in drop]
    num_cols = [c for c in df.columns if df[c].dtype != "O" and c not in drop]

    # Treat low-cardinality numerics as categorical
    num_but_cat = [c for c in num_cols if df[c].nunique() < thr]
    cat_cols += num_but_cat
    num_cols = [c for c in num_cols if c not in num_but_cat]

    return cat_cols, num_cols


# ── Encoders ──────────────────────────────────────────────────────────────────

def fit_encoders(train: pd.DataFrame, cfg: dict) -> dict:
    """Fit all encoders on Train only. Returns dict of fitted transformers."""
    f = cfg["features"]
    encoders = {}

    # OrdinalEncoder
    ord_cats = [f["ordinal_categories"][c] for c in f["ordinal_cols"]]
    enc_ord = OrdinalEncoder(
        categories=ord_cats,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    enc_ord.fit(train[f["ordinal_cols"]])
    encoders["ordinal"] = enc_ord
    logger.info(f"OrdinalEncoder fit on: {f['ordinal_cols']}")

    # OneHotEncoder
    ohe_cols = [c for c in f["onehot_cols"] if c in train.columns]
    enc_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    enc_ohe.fit(train[ohe_cols])
    encoders["ohe"] = enc_ohe
    encoders["ohe_cols"] = ohe_cols
    logger.info(f"OneHotEncoder fit on: {ohe_cols}")

    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict, cfg: dict) -> pd.DataFrame:
    """Apply pre-fitted encoders to any split."""
    df = df.copy()
    f = cfg["features"]

    # Ordinal
    df[f["ordinal_cols"]] = encoders["ordinal"].transform(df[f["ordinal_cols"]])

    # OHE
    ohe_cols = encoders["ohe_cols"]
    encoded = encoders["ohe"].transform(df[ohe_cols])
    new_cols = encoders["ohe"].get_feature_names_out(ohe_cols)
    encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    return df


def fit_scaler(train: pd.DataFrame, num_cols: list) -> StandardScaler:
    """Fit StandardScaler on Train numerical columns."""
    scaler = StandardScaler()
    scaler.fit(train[num_cols])
    logger.info(f"StandardScaler fit on {len(num_cols)} numerical columns")
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, num_cols: list) -> pd.DataFrame:
    """Apply pre-fitted scaler."""
    df = df.copy()
    cols_present = [c for c in num_cols if c in df.columns]
    df[cols_present] = scaler.transform(df[cols_present])
    return df


# ── X/y Split ─────────────────────────────────────────────────────────────────

def get_Xy(df: pd.DataFrame, cfg: dict, task: str = "regression") -> tuple:
    """Return X and y for a given task."""
    if task == "regression":
        target = cfg["features"]["target_regression"]
    else:
        target = cfg["features"]["target_classification"]

    X = df.drop(columns=[
        cfg["features"]["target_regression"],
        cfg["features"]["target_classification"]
    ], errors="ignore")
    y = df[target]
    return X, y


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(raw_path: str = None) -> dict:
    """
    Full preprocessing pipeline.
    Returns dict with all splits (X_train, y_train_reg, y_train_clf, etc.)
    and saves all artifacts to disk.
    """
    cfg = load_config()
    raw_path = raw_path or (ROOT / cfg["data"]["raw_path"])

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE START")
    logger.info("=" * 60)

    # ── Load ──
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Save reference snapshot (for drift detection) ──
    ref_path = ROOT / cfg["data"]["reference_path"] / "reference_snapshot.parquet"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ref_path, index=False)
    logger.info(f"Reference snapshot saved → {ref_path}")

    # ── Drop columns not needed at training time ──
    drop_cols = [c for c in cfg["features"]["drop_cols"] if c in df.columns]
    df_model = df.drop(columns=drop_cols)

    # ── Feature engineering ──
    df_model = feature_engineering(df_model)
    logger.info("Feature engineering done")

    # ── Nested split ──
    logger.info("Nested 5-partition split:")
    splits = nested_split(df_model, cfg)

    # ── Column types (fit on train) ──
    _, num_cols = create_col_types(splits["train"], cfg)

    # ── Fit transformers on train only ──
    encoders = fit_encoders(splits["train"], cfg)

    # ── Apply encoders to all splits ──
    processed = {}
    for name, split in splits.items():
        processed[name] = apply_encoders(split, encoders, cfg)

    # ── Fit scaler on encoded train ──
    # Recompute num_cols after encoding (OHE adds columns)
    targets = [cfg["features"]["target_regression"], cfg["features"]["target_classification"]]
    num_cols_final = [
        c for c in processed["train"].columns
        if processed["train"][c].dtype != "O"
        and c not in targets
    ]
    scaler = fit_scaler(processed["train"], num_cols_final)

    # ── Apply scaler ──
    for name in processed:
        processed[name] = apply_scaler(processed[name], scaler, num_cols_final)

    # ── Save artifacts ──
    artifacts_dir = ROOT / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    save_artifact(encoders, artifacts_dir / "encoders.joblib")
    save_artifact(scaler, artifacts_dir / "scaler.joblib")
    save_artifact(num_cols_final, artifacts_dir / "num_cols.joblib")

    # ── Save processed splits ──
    proc_dir = ROOT / cfg["data"]["processed_path"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    for name, split in processed.items():
        split.to_parquet(proc_dir / f"{name}.parquet", index=False)
    logger.info(f"Processed splits saved → {proc_dir}")

    # ── Build output dict ──
    result = {}
    for name, split in processed.items():
        result[f"X_{name}"], result[f"y_{name}_reg"]  = get_Xy(split, cfg, "regression")
        _,                    result[f"y_{name}_clf"]  = get_Xy(split, cfg, "classification")

    result["cfg"]           = cfg
    result["encoders"]      = encoders
    result["scaler"]        = scaler
    result["num_cols"]      = num_cols_final
    result["feature_names"] = list(result["X_train"].columns)

    logger.info("PREPROCESSING COMPLETE")
    return result


if __name__ == "__main__":
    run_preprocessing()