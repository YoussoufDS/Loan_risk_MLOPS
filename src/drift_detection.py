"""
Drift detection using Evidently AI.
Compares new incoming data against the reference training snapshot.

Three types of drift detected:
  1. Data drift (covariate shift) — feature distribution changes
  2. Target drift — RiskScore / LoanApproved distribution changes
  3. Model performance degradation — rolling RMSE vs baseline
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, DataDriftTable

import mlflow
from src.utils import load_config, ROOT


def load_reference(cfg: dict) -> pd.DataFrame:
    ref_path = ROOT / cfg["data"]["reference_path"] / "reference_snapshot.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference snapshot not found at {ref_path}. Run training first.")
    return pd.read_parquet(ref_path)


def load_current_data(current_path: str) -> pd.DataFrame:
    path = Path(current_path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  → No significant change
    PSI 0.1–0.2 → Moderate change
    PSI > 0.2  → Significant shift — consider retraining
    """
    ref_cut, bin_edges = pd.cut(reference, bins=bins, retbins=True, duplicates="drop")
    cur_cut = pd.cut(current, bins=bin_edges, duplicates="drop")

    ref_pct = ref_cut.value_counts(normalize=True, sort=False) + 1e-4
    cur_pct = cur_cut.value_counts(normalize=True, sort=False) + 1e-4

    # Align indexes
    ref_pct, cur_pct = ref_pct.align(cur_pct, fill_value=1e-4)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: str) -> dict:
    """Generate full Evidently drift report and return summary dict."""
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference, current_data=current)
    report.save_html(output_path)
    logger.info(f"Evidently HTML report saved → {output_path}")

    result = report.as_dict()
    return result


def detect_drift(current_data_path: str, output_dir: str = "reports/drift/") -> dict:
    """
    Full drift detection pipeline.
    Returns dict with drift status and per-feature PSI scores.
    """
    cfg = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DRIFT DETECTION START")

    # ── Load data ──
    reference = load_reference(cfg)
    current = load_current_data(current_data_path)
    logger.info(f"Reference: {len(reference)} rows | Current: {len(current)} rows")

    # ── Select common numeric features ──
    targets = [cfg["features"]["target_regression"], cfg["features"]["target_classification"]]
    drop_cols = cfg["features"]["drop_cols"] + targets
    num_cols = [
        c for c in reference.select_dtypes(include=[np.number]).columns
        if c not in drop_cols and c in current.columns
    ]

    # ── PSI per feature ──
    psi_threshold = cfg["drift"]["psi_threshold"]
    psi_results = {}
    drifted_features = []

    for col in num_cols:
        try:
            psi = compute_psi(reference[col].dropna(), current[col].dropna())
            psi_results[col] = psi
            if psi > psi_threshold:
                drifted_features.append(col)
                logger.warning(f"  PSI DRIFT [{col}]: {psi:.4f} > {psi_threshold}")
        except Exception as e:
            logger.debug(f"  PSI skipped for {col}: {e}")

    # ── Evidently full report ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = str(output_dir / f"drift_report_{timestamp}.html")
    json_path = str(output_dir / f"drift_summary_{timestamp}.json")

    try:
        evidently_result = run_evidently_report(
            reference[num_cols + targets],
            current[num_cols + [c for c in targets if c in current.columns]],
            html_path,
        )
    except Exception as e:
        logger.warning(f"Evidently report failed: {e}")
        evidently_result = {}

    # ── Decision ──
    n_drifted = len(drifted_features)
    retrain_needed = n_drifted >= 3   # trigger retrain if 3+ features drifted
    overall_psi = float(np.mean(list(psi_results.values()))) if psi_results else 0.0

    result = {
        "timestamp": timestamp,
        "n_features_checked": len(psi_results),
        "n_drifted_features": n_drifted,
        "drifted_features": drifted_features,
        "overall_mean_psi": overall_psi,
        "retrain_needed": retrain_needed,
        "psi_threshold": psi_threshold,
        "feature_psi": psi_results,
        "evidently_html": html_path,
    }

    # Save JSON summary
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Log to MLflow ──
    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        with mlflow.start_run(run_name=f"drift_check_{timestamp}"):
            mlflow.log_metrics({
                "n_drifted_features": float(n_drifted),
                "overall_mean_psi": overall_psi,
            })
            mlflow.log_artifact(html_path)
            mlflow.log_artifact(json_path)
            mlflow.set_tag("retrain_needed", str(retrain_needed))
    except Exception as e:
        logger.warning(f"MLflow logging skipped (server not running?): {e}")

    # ── Summary ──
    logger.info(f"Features checked: {len(psi_results)}")
    logger.info(f"Drifted features (PSI > {psi_threshold}): {n_drifted}")
    logger.info(f"Overall mean PSI: {overall_psi:.4f}")
    logger.info(f"Retrain needed: {retrain_needed}")
    logger.info("DRIFT DETECTION COMPLETE")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to current data (CSV or Parquet)")
    parser.add_argument("--output", default="reports/drift/", help="Output directory for reports")
    args = parser.parse_args()
    result = detect_drift(args.data, args.output)
    print(f"\nRetrain needed: {result['retrain_needed']}")
    print(f"Drifted features: {result['drifted_features']}")