"""
Pydantic models for loan evaluation system.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class LoanApplication(BaseModel):
    """Main loan application data model."""
    
    # Applicant Information
    applicant_name: str
    applicant_email: str
    applicant_phone: str
    date_of_birth: str  # Format: YYYY-MM-DD
    
    # Loan Details
    requested_amount: float = Field(gt=0, description="Requested loan amount in USD")
    loan_purpose: Literal[
        "Home Purchase",
        "Refinancing",
        "Home Improvement",
        "Debt Consolidation",
        "Business",
        "Auto Purchase",
        "Education",
        "Other"
    ]
    loan_term_months: int = Field(default=360, ge=12, le=480)
    
    # Employment & Income
    employment_status: Literal["Employed", "Self-Employed", "Unemployed", "Retired"]
    current_employer: str
    employment_years: float = Field(ge=0)
    annual_income: float = Field(gt=0)
    secondary_income: Optional[float] = None
    income_source: str  # e.g., W2, Self-Employment, Investment, etc.
    
    # Credit Profile
    credit_score: int = Field(ge=300, le=850)
    credit_inquiries_6_months: int = Field(default=0, ge=0)
    existing_debt: float = Field(default=0, ge=0)
    
    # Financial Details
    down_payment: Optional[float] = None
    collateral_value: Optional[float] = None
    liquid_assets: float = Field(default=0, ge=0)
    bankruptcy_history: Optional[str] = None
    
    # Documentation
    documents_provided: List[str] = Field(
        default_factory=list,
        description="List of document types: pay_stubs, tax_returns, bank_statements, etc."
    )
    
    # Additional Information
    address: str
    years_at_address: float = Field(default=0, ge=0)
    notes: Optional[str] = None
    submission_date: datetime = Field(default_factory=datetime.now)


class ReviewResult(BaseModel):
    """Individual reviewer assessment."""
    
    reviewer_name: str
    score: int = Field(ge=0, le=100, description="Overall score 0-100")
    strengths: List[str]
    weaknesses: List[str]
    risks: List[str]
    recommended_action: Literal["approve", "conditional_approval", "deny", "request_more_info", "manual_review"]
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Reviewer-specific fields (optional)
    fraud_risk_level: Optional[Literal["low", "medium", "high"]] = None
    fraud_risk_score: Optional[int] = Field(default=None, ge=0, le=10)
    suspicious_items: Optional[List[str]] = None
    investigation_needed: Optional[List[str]] = None
    
    compliance_issues: Optional[List[str]] = None
    documentation_status: Optional[Literal["complete", "incomplete", "needs_verification"]] = None
    
    recommended_terms: Optional[Dict[str, Any]] = None
    conditions_for_approval: Optional[List[str]] = None
    required_documents: Optional[List[str]] = None
    overall_rationale: Optional[str] = None


class EvaluationResult(BaseModel):
    """Complete evaluation result from all reviewers."""
    
    application_id: str
    applicant_name: str
    submission_date: datetime
    evaluation_date: datetime = Field(default_factory=datetime.now)
    
    # Individual Reviews
    reviews: List[ReviewResult]
    
    # Synthesis
    final_recommendation: Literal["approve", "conditional_approval", "deny", "request_more_info"]
    overall_score: float  # Weighted average
    decision_rationale: str
    
    # Summary metrics
    avg_confidence: float
    fraud_risk_summary: Optional[str] = None
    compliance_status: Optional[str] = None
    
    # Recommended Terms (if approved/conditional)
    recommended_interest_rate: Optional[float] = None
    recommended_loan_amount: Optional[float] = None
    special_conditions: Optional[List[str]] = None
    
    # Meta
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    agents_involved: List[str] = Field(default_factory=list)
    total_tokens_used: Optional[int] = None


class CreditAnalysisDetail(BaseModel):
    """Detailed credit analysis metrics."""
    
    credit_score: int
    credit_score_trend: Literal["improving", "stable", "declining"]
    recent_inquiries: int = 0
    delinquencies_30: int = 0
    delinquencies_60: int = 0
    delinquencies_90_plus: int = 0
    debt_to_income_ratio: float = Field(ge=0, le=1)
    loan_to_value_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    income_stability_score: int = Field(ge=0, le=100)
    payment_history_score: int = Field(ge=0, le=100)


class FraudAnalysisDetail(BaseModel):
    """Detailed fraud analysis findings."""
    
    fraud_risk_score: int = Field(ge=0, le=100)
    document_inconsistencies: List[str] = Field(default_factory=list)
    income_verification_issues: List[str] = Field(default_factory=list)
    employment_verification_status: Literal["verified", "needs_review", "failed"]
    anomaly_flags: List[str] = Field(default_factory=list)
    manual_review_required: bool = False


class ComplianceAnalysisDetail(BaseModel):
    """Detailed compliance assessment."""
    
    kyc_status: Literal["complete", "incomplete", "needs_verification"]
    aml_status: Literal["clear", "review_needed", "flagged"]
    pep_screening_status: Literal["clear", "potential_match", "flagged"]
    documentation_completeness: float = Field(ge=0, le=1)
    regulatory_issues: List[str] = Field(default_factory=list)
    compliance_risk_level: Literal["low", "medium", "high"]
