"""
WeightedEnsemble — module séparé pour permettre la désérialisation joblib.
Doit être importé avant de charger un modèle sérialisé avec joblib.
"""

import numpy as np


class WeightedEnsemble:
    """Combines model predictions with Hill Climbing weights."""

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