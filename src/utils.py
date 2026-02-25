"""Shared utilities — logging, config loading, paths."""

import yaml
import joblib
from pathlib import Path
from loguru import logger
import sys

# ── Logger setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}", level="INFO")
logger.add("logs/pipeline.log", rotation="10 MB", retention="30 days", level="DEBUG")

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """Load central config.yaml."""
    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_artifact(obj, path: str) -> None:
    """Save a Python object with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.debug(f"Artifact saved → {path}")


def load_artifact(path: str):
    """Load a joblib artifact."""
    return joblib.load(path)