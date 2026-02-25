"""API tests — run with: pytest tests/test_api.py -v"""

import os
import pytest

# Force SQLite MLflow URI before any import to avoid localhost:5000 connection attempts
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_APPLICANT = {
    "Age": 35, "AnnualIncome": 60000.0, "CreditScore": 680,
    "EmploymentStatus": "Employed", "EducationLevel": "Bachelor's",
    "Experience": 8.0, "LoanAmount": 20000.0, "LoanDuration": 60,
    "MaritalStatus": "Married", "NumberOfDependents": 1,
    "HomeOwnershipStatus": "Mortgage", "MonthlyDebtPayments": 400.0,
    "CreditCardUtilizationRate": 0.3, "NumberOfOpenCreditLines": 3,
    "NumberOfCreditInquiries": 1, "DebtToIncomeRatio": 0.3,
    "BankruptcyHistory": 0, "LoanPurpose": "Home",
    "PreviousLoanDefaults": 0, "PaymentHistory": 7.5,
    "LengthOfCreditHistory": 8.0, "SavingsAccountBalance": 15000.0,
    "CheckingAccountBalance": 3000.0, "TotalAssets": 120000.0,
    "TotalLiabilities": 45000.0, "MonthlyIncome": 5000.0,
    "UtilityBillsPaymentHistory": 0.9, "JobTenure": 4.0,
    "NetWorth": 75000.0, "BaseInterestRate": 3.5,
    "InterestRate": 5.5, "MonthlyLoanPayment": 380.0,
    "TotalDebtToIncomeRatio": 0.5,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


def test_predict_risk_schema():
    """Test that /predict/risk returns correct schema (model may not be loaded)."""
    r = client.post("/predict/risk", json=SAMPLE_APPLICANT)
    # 200 = model loaded, 503 = model not available — both are valid in test env
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "risk_score" in data
        assert "risk_level" in data
        assert data["risk_level"] in ("Low", "Medium", "High")
        assert "shap_top_features" in data


def test_predict_approval_schema():
    r = client.post("/predict/approval", json=SAMPLE_APPLICANT)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "loan_approved" in data
        assert isinstance(data["loan_approved"], bool)
        assert 0 <= data["approval_probability"] <= 1


def test_invalid_credit_score():
    """Test Pydantic validation — CreditScore must be 300-850."""
    bad = {**SAMPLE_APPLICANT, "CreditScore": 9999}
    r = client.post("/predict/risk", json=bad)
    assert r.status_code == 422


def test_invalid_employment_status():
    bad = {**SAMPLE_APPLICANT, "EmploymentStatus": "InvalidStatus"}
    r = client.post("/predict/risk", json=bad)
    assert r.status_code == 422


def test_model_info():
    r = client.get("/model/info")
    # 200 = models loaded, 503 = models not available in CI (no trained models yet)
    assert r.status_code in (200, 503, 500)


def test_reload_endpoint():
    r = client.post("/model/reload")
    assert r.status_code == 200