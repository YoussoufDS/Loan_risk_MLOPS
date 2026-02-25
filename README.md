# 🏦 Loan Risk MLOps

Pipeline MLOps complet pour la prédiction de risque de prêt.

## Stack
`LightGBM · XGBoost · CatBoost · Optuna · MLflow · FastAPI · Streamlit · GitHub Actions · Evidently`

## Démarrage rapide

```bash
# 1. Cloner et installer
git clone <repo-url>
cd loan-risk-mlops
make setup

# 2. Placer le dataset
cp /chemin/vers/Loan.csv data/raw/

# 3. Démarrer MLflow (terminal 1)
make mlflow        # → http://localhost:5000

# 4. Entraîner les modèles (terminal 2)
make train

# 5. Démarrer l'API (terminal 3)
make api           # → http://localhost:8000/docs

# 6. Démarrer le frontend (terminal 4)
make frontend      # → http://localhost:8501
```

## Architecture

```
data/           → Raw + processed splits + reference snapshot
src/
  preprocessing.py    → Feature engineering + nested 5-split + encodeurs
  optimize.py         → Optuna (Val-B) + Hill Climbing (Val-C)
  train.py            → Orchestrateur principal + MLflow logging
  evaluate.py         → Métriques + SHAP + plots
  drift_detection.py  → Evidently + PSI + auto-trigger retrain
api/
  main.py             → FastAPI (6 endpoints)
  predict.py          → Chargement modèle + inférence
  schemas.py          → Validation Pydantic
frontend/
  app.py              → Streamlit 4 pages
.github/workflows/
  train.yml           → Entraînement manuel/push
  retrain.yml         → CRON hebdo + drift trigger
  drift_check.yml     → CRON quotidien
  deploy.yml          → Validation + déploiement
```

## Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Statut API + versions modèles |
| POST | `/predict/risk` | RiskScore + SHAP |
| POST | `/predict/approval` | LoanApproved + proba + SHAP |
| POST | `/predict/batch` | Upload CSV → CSV enrichi |
| GET | `/model/info` | Métadonnées modèles actifs |
| POST | `/model/reload` | Rechargement depuis MLflow Registry |

## Design décisions clés

**Nested split 5 partitions** — résout la contamination du validation set :
- Train 60% → entraînement
- Val-A 10% → early stopping uniquement
- Val-B 10% → Optuna uniquement
- Val-C 10% → Hill Climbing uniquement
- Test 10%  → évaluation finale, touché une seule fois

**Promotion automatique** — si RMSE s'améliore de >1% vs Production → auto-promote

**Drift → Retrain** — si PSI > 0.2 sur ≥3 features → déclenche retrain.yml automatiquement