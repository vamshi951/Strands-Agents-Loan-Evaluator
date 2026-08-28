#!/usr/bin/env python3
"""
Demonstration script showing the loan evaluation workflow.
This shows how data flows through the system without requiring Strands/AWS.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loan_evaluator import (
    LoanApplication,
    ReviewResult,
    EvaluationResult,
    CreditAnalysisDetail,
    FraudAnalysisDetail,
    ComplianceAnalysisDetail,
)

print("\n" + "=" * 70)
print("LOAN EVALUATOR - SYSTEM DEMONSTRATION")
print("=" * 70)

# Load a sample application
print("\n[STEP 1] Loading Loan Application...")
sample_path = project_root / "loan_evaluator" / "sample_data" / "loan_application_1.json"

with open(sample_path) as f:
    app_data = json.load(f)

app = LoanApplication(**app_data)

print(f"""
Applicant Profile:
  Name: {app.applicant_name}
  Email: {app.applicant_email}
  Age: ~{datetime.now().year - int(app.date_of_birth.split('-')[0])} years old
  
Loan Details:
  Requested Amount: ${app.requested_amount:,.2f}
  Purpose: {app.loan_purpose}
  Term: {app.loan_term_months} months
  
Financial Profile:
  Credit Score: {app.credit_score}
  Annual Income: ${app.annual_income:,.2f}
  Existing Debt: ${app.existing_debt:,.2f}
  Liquid Assets: ${app.liquid_assets:,.2f}
  
Employment:
  Status: {app.employment_status}
  Employer: {app.current_employer}
  Tenure: {app.employment_years} years
""")

# Calculate key metrics
print("[STEP 2] Computing Financial Metrics...")
debt_to_income = (app.existing_debt * 12 / app.annual_income) if app.annual_income else 0
loan_to_value = (app.requested_amount / app.collateral_value * 100) if app.collateral_value else 0
monthly_income = app.annual_income / 12
monthly_debt = app.existing_debt / 12

print(f"""
Key Metrics:
  Debt-to-Income Ratio: {debt_to_income:.2%}
  Loan-to-Value Ratio: {loan_to_value:.1f}%
  Monthly Income: ${monthly_income:,.2f}
  Monthly Debt: ${monthly_debt:,.2f}
  Available Monthly: ${monthly_income - monthly_debt:,.2f}
""")

# Simulate agent evaluations
print("[STEP 3] Running Agent Evaluations (Simulated)...")
print("\nNote: These are simulated evaluations based on rules.")
print("Full AI evaluation requires AWS Bedrock + Strands package.\n")

# Credit Analyst Assessment
print("👨‍💼 CREDIT RISK ANALYST")
print("-" * 50)

credit_score_rating = "Excellent" if app.credit_score >= 750 else "Good" if app.credit_score >= 700 else "Fair" if app.credit_score >= 650 else "Poor"
dti_rating = "Good" if debt_to_income < 0.43 else "Elevated" if debt_to_income < 0.50 else "High"
ltv_rating = "Conservative" if loan_to_value < 80 else "Moderate" if loan_to_value < 90 else "Aggressive"

credit_score = 75
if app.credit_score >= 750:
    credit_score += 15
elif app.credit_score >= 700:
    credit_score += 10

if debt_to_income < 0.36:
    credit_score += 10
elif debt_to_income > 0.50:
    credit_score -= 10

if app.employment_years >= 5:
    credit_score += 5

credit_review = ReviewResult(
    reviewer_name="credit_risk_analyst",
    score=min(100, credit_score),
    strengths=[
        f"Strong credit score ({app.credit_score}): {credit_score_rating}",
        f"Stable employment: {app.employment_years} years",
        f"Good liquid assets: ${app.liquid_assets:,.0f}"
    ],
    weaknesses=[
        f"Debt-to-income ratio: {debt_to_income:.2%} ({dti_rating})" if debt_to_income > 0.36 else "Low debt obligations"
    ] if debt_to_income > 0.36 else [],
    risks=[
        "Recent credit inquiries" if app.credit_inquiries_6_months > 2 else "Minimal recent inquiries"
    ],
    recommended_action="approve" if credit_score >= 75 else "request_more_info",
    confidence=0.85 if credit_score >= 75 else 0.70,
)

print(f"Score: {credit_review.score}/100")
print(f"Recommendation: {credit_review.recommended_action}")
print(f"Confidence: {credit_review.confidence:.0%}\n")

# Compliance Officer Assessment
print("⚖️  COMPLIANCE OFFICER")
print("-" * 50)

doc_status = "complete" if len(app.documents_provided) >= 4 else "incomplete"
doc_count = len(app.documents_provided)
compliance_score = 70 + (doc_count * 5)

compliance_review = ReviewResult(
    reviewer_name="compliance_officer",
    score=min(100, compliance_score),
    strengths=[
        f"Documentation: {doc_count} documents provided",
        "Employment verification available",
        "No bankruptcy history reported"
    ],
    weaknesses=[] if doc_status == "complete" else ["Additional documentation needed"],
    risks=[],
    documentation_status=doc_status,
    recommended_action="approve" if compliance_score >= 75 else "request_more_info",
    confidence=0.90,
    compliance_issues=[] if compliance_score >= 75 else ["Incomplete documentation package"],
)

print(f"Score: {compliance_review.score}/100")
print(f"Documentation Status: {doc_status}")
print(f"Recommendation: {compliance_review.recommended_action}\n")

# Fraud Detection Assessment
print("🔍 FRAUD DETECTION SPECIALIST")
print("-" * 50)

fraud_risk_score = 2  # low base score
anomalies = []

if app.credit_inquiries_6_months > 5:
    fraud_risk_score += 2
    anomalies.append("Multiple recent inquiries")

if app.employment_years < 1 and app.employment_status == "Employed":
    fraud_risk_score += 3
    anomalies.append("Recent employment change")

if len(app.documents_provided) < 2:
    fraud_risk_score += 2
    anomalies.append("Limited documentation provided")

fraud_risk_level = "low" if fraud_risk_score < 3 else "medium" if fraud_risk_score < 6 else "high"
fraud_review = ReviewResult(
    reviewer_name="fraud_detection_specialist",
    score=85 - (fraud_risk_score * 5),
    fraud_risk_level=fraud_risk_level,
    fraud_risk_score=fraud_risk_score,
    strengths=["Consistent income documentation", "Clear employment history"],
    weaknesses=[] if fraud_risk_score < 3 else anomalies,
    risks=anomalies if anomalies else ["None identified"],
    suspicious_items=anomalies,
    investigation_needed=[] if fraud_risk_score < 3 else anomalies,
    recommended_action="approve" if fraud_risk_score < 3 else "request_more_info",
    confidence=0.90,
)

print(f"Score: {fraud_review.score}/100")
print(f"Fraud Risk Level: {fraud_risk_level.upper()}")
print(f"Fraud Risk Score: {fraud_risk_score}/10")
print(f"Recommendation: {fraud_review.recommended_action}\n")

# Loan Officer Final Decision
print("✍️  LOAN OFFICER (FINAL DECISION)")
print("-" * 50)

reviews = [credit_review, compliance_review, fraud_review]
overall_score = sum(r.score for r in reviews) / len(reviews)

recommendation = "approve" if overall_score >= 75 and fraud_risk_score < 3 else "conditional_approval" if overall_score >= 70 else "request_more_info"

# Recommend interest rate based on credit score
base_rate = 0.065  # 6.5%
credit_adjustment = (750 - app.credit_score) * 0.0002  # 0.02% per credit point below 750
recommended_rate = base_rate + credit_adjustment

loan_officer_review = ReviewResult(
    reviewer_name="loan_officer",
    score=int(overall_score),
    strengths=[
        f"Overall strong profile with average score {overall_score:.0f}/100",
        "Positive specialist reviews across all dimensions"
    ],
    weaknesses=[],
    risks=[],
    recommended_action=recommendation,
    recommended_terms={
        "interest_rate": recommended_rate,
        "loan_amount": app.requested_amount,
        "loan_term_months": app.loan_term_months,
        "special_conditions": [] if recommendation == "approve" else ["Verify recent employment"]
    },
    conditions_for_approval=[] if recommendation == "approve" else ["Employment verification letter"],
    confidence=0.88,
)

print(f"Final Score: {loan_officer_review.score}/100")
print(f"Recommendation: {recommendation.upper()}")
print(f"Recommended Interest Rate: {recommended_rate:.3%}")
print(f"Loan Amount: ${app.requested_amount:,.2f}")
print(f"Term: {app.loan_term_months} months")
print(f"Monthly Payment (est.): ${(app.requested_amount * (recommended_rate/12) / (1 - (1 + recommended_rate/12)**(-app.loan_term_months))):,.2f}\n")

# Create Final Evaluation Result
print("[STEP 4] Compiling Final Evaluation Result...")
print("-" * 50)

evaluation = EvaluationResult(
    application_id=f"LOAN-{app.applicant_name.replace(' ', '-')}",
    applicant_name=app.applicant_name,
    submission_date=app.submission_date,
    reviews=reviews + [loan_officer_review],
    final_recommendation=recommendation,
    overall_score=overall_score,
    decision_rationale=f"Applicant meets lending criteria with {overall_score:.0f}/100 overall score. All specialists recommend {recommendation}.",
    avg_confidence=sum(r.confidence for r in reviews) / len(reviews),
    fraud_risk_summary=f"{fraud_risk_level.upper()} (score: {fraud_risk_score}/10)",
    compliance_status=doc_status.upper(),
    recommended_interest_rate=recommended_rate,
    recommended_loan_amount=app.requested_amount,
    agents_involved=["credit_risk_analyst", "compliance_officer", "fraud_detection_specialist", "loan_officer"],
)

print(f"✅ Evaluation compiled successfully")
print(f"   Application ID: {evaluation.application_id}")
print(f"   Final Recommendation: {evaluation.final_recommendation}")
print(f"   Overall Score: {evaluation.overall_score:.1f}/100\n")

# Export to JSON
print("[STEP 5] Exporting Results...")
print("-" * 50)

export_file = project_root / f"evaluation_{app.applicant_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(export_file, "w") as f:
    f.write(evaluation.model_dump_json(indent=2))

print(f"✅ Results exported to: {export_file.name}")
print(f"   File size: {export_file.stat().st_size} bytes\n")

print("=" * 70)
print("✅ DEMONSTRATION COMPLETE")
print("=" * 70)
print(f"""
Next Steps:
1. Install dependencies: pip install -r requirements.txt
2. Configure AWS credentials: aws configure
3. Run notebook: jupyter notebook loan_evaluator/evaluator_v1.ipynb
4. Or run Python: python loan_evaluator/loan_evaluator.py

For more information, see: README.md
""")
