# ── Loan Risk MLOps — Makefile ─────────────────────────────────────────────
# Usage: make <target>

.PHONY: help setup mlflow api frontend train test drift all

help:
	@echo ""
	@echo "  Loan Risk MLOps — Commandes disponibles"
	@echo "  ─────────────────────────────────────────"
	@echo "  make setup     → Installer les dépendances"
	@echo "  make mlflow    → Démarrer MLflow UI (http://localhost:5000)"
	@echo "  make train     → Lancer l'entraînement complet"
	@echo "  make api       → Démarrer l'API FastAPI (http://localhost:8000)"
	@echo "  make frontend  → Démarrer Streamlit (http://localhost:8501)"
	@echo "  make test      → Lancer les tests pytest"
	@echo "  make drift     → Lancer la détection de drift"
	@echo "  make all       → Tout démarrer (mlflow + api + frontend)"
	@echo ""

setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/reference models/artifacts \
	         reports/drift mlruns logs
	@echo "✅ Setup complete. Place Loan.csv in data/raw/"

mlflow:
	mlflow server \
	  --backend-store-uri sqlite:///mlruns/mlflow.db \
	  --default-artifact-root ./mlruns \
	  --host 127.0.0.1 \
	  --port 5000

train:
	python -m src.train --data data/raw/Loan.csv --trigger manual

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	streamlit run frontend/app.py --server.port 8501

test:
	pytest tests/ -v --tb=short

drift:
	python src/drift_detection.py --data data/raw/Loan.csv --output reports/drift/

# Launch all three services in background (dev mode)
all:
	@echo "Starting MLflow, API, and Streamlit..."
	@make mlflow &
	@sleep 3
	@make api &
	@sleep 3
	@make frontend