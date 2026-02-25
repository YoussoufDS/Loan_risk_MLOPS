"""Pydantic schemas for API input validation and response formatting."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


class LoanApplicant(BaseModel):
    """Input schema — all 35 original features (post feature-engineering done server-side)."""

    Age: int                             = Field(..., ge=18, le=100)
    AnnualIncome: float                  = Field(..., ge=0)
    CreditScore: int                     = Field(..., ge=300, le=850)
    EmploymentStatus: Literal[
        "Employed", "Self-Employed", "Unemployed"
    ]
    EducationLevel: str
    Experience: float                    = Field(..., ge=0)
    LoanAmount: float                    = Field(..., ge=0)
    LoanDuration: int                    = Field(..., ge=1, le=360)
    MaritalStatus: str
    NumberOfDependents: int              = Field(..., ge=0, le=20)
    HomeOwnershipStatus: str
    MonthlyDebtPayments: float           = Field(..., ge=0)
    CreditCardUtilizationRate: float     = Field(..., ge=0.0, le=1.0)
    NumberOfOpenCreditLines: int         = Field(..., ge=0)
    NumberOfCreditInquiries: int         = Field(..., ge=0)
    DebtToIncomeRatio: float             = Field(..., ge=0)
    BankruptcyHistory: int               = Field(..., ge=0, le=1)
    LoanPurpose: str
    PreviousLoanDefaults: int            = Field(..., ge=0)
    PaymentHistory: float
    LengthOfCreditHistory: float         = Field(..., ge=0)
    SavingsAccountBalance: float         = Field(..., ge=0)
    CheckingAccountBalance: float        = Field(..., ge=0)
    TotalAssets: float                   = Field(..., ge=0)
    TotalLiabilities: float              = Field(..., ge=0)
    MonthlyIncome: float                 = Field(..., ge=0)
    UtilityBillsPaymentHistory: float
    JobTenure: float                     = Field(..., ge=0)
    NetWorth: float
    BaseInterestRate: float              = Field(..., ge=0)
    InterestRate: float                  = Field(..., ge=0)
    MonthlyLoanPayment: float            = Field(..., ge=0)
    TotalDebtToIncomeRatio: float        = Field(..., ge=0)

    model_config = {"extra": "allow"}


class RiskPredictionResponse(BaseModel):
    risk_score: float
    risk_level: Literal["Low", "Medium", "High"]
    model_version: str
    shap_top_features: list[dict]


class ApprovalPredictionResponse(BaseModel):
    loan_approved: bool
    approval_probability: float
    model_version: str
    shap_top_features: list[dict]


class HealthResponse(BaseModel):
    status: str
    regression_model_version: Optional[str]
    classification_model_version: Optional[str]
    mlflow_uri: str


class ModelInfoResponse(BaseModel):
    regression: dict
    classification: dict