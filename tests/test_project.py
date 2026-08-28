#!/usr/bin/env python3
"""
Test script for Loan Evaluator project.
Tests data models, configuration, and project structure.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project package to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("LOAN EVALUATOR - PROJECT TEST SUITE")
print("=" * 70)

# Test 1: Import models
print("\n[TEST 1] Importing data models...")
try:
    from loan_evaluator import (
        LoanApplication,
        ReviewResult,
        EvaluationResult,
        CreditAnalysisDetail,
        FraudAnalysisDetail,
        ComplianceAnalysisDetail,
    )
    print("✅ All models imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Load and validate sample data
print("\n[TEST 2] Loading sample loan applications...")
sample_files = list((project_root / "loan_evaluator" / "sample_data").glob("*.json"))

if not sample_files:
    print("❌ No sample data files found")
    sys.exit(1)

for sample_file in sorted(sample_files):
    try:
        with open(sample_file, "r") as f:
            data = json.load(f)
        
        # Validate against model
        app = LoanApplication(**data)
        print(f"✅ {sample_file.name}: {app.applicant_name} - ${app.requested_amount:,}")
    except Exception as e:
        print(f"❌ {sample_file.name}: {e}")
        sys.exit(1)

# Test 3: Create sample review results
print("\n[TEST 3] Creating sample review results...")
try:
    review = ReviewResult(
        reviewer_name="test_reviewer",
        score=75,
        strengths=["Good credit history", "Stable employment"],
        weaknesses=["High debt-to-income ratio"],
        risks=["Recent job change"],
        recommended_action="conditional_approval",
        confidence=0.85,
    )
    print(f"✅ Created review: {review.reviewer_name} (Score: {review.score}/100)")
except Exception as e:
    print(f"❌ Review creation failed: {e}")
    sys.exit(1)

# Test 4: Check prompt files
print("\n[TEST 4] Checking prompt templates...")
prompts_dir = project_root / "loan_evaluator" / "prompts"
expected_prompts = [
    "credit_analyst_prompt.txt",
    "compliance_prompt.txt",
    "fraud_detection_prompt.txt",
    "loan_officer_prompt.txt",
]

for prompt_file in expected_prompts:
    filepath = prompts_dir / prompt_file
    if filepath.exists():
        size = filepath.stat().st_size
        print(f"✅ {prompt_file}: {size} bytes")
    else:
        print(f"❌ {prompt_file}: NOT FOUND")
        sys.exit(1)

# Test 5: Normalize alternate LLM review response shapes
print("\n[TEST 5] Normalizing alternate review JSON...")
try:
    from loan_evaluator.loan_evaluator import LoanEvaluator

    normalized = LoanEvaluator._normalize_review_json(
        {
            "reviewer_name": "loan_officer",
            "score": 82,
            "final_recommendation": "approve",
            "strengths": ["Strong reserves"],
            "weaknesses": [],
            "risks": [],
            "confidence": 0.99,
        },
        "loan_officer",
    )
    normalized_review = ReviewResult(**normalized)
    assert normalized_review.recommended_action == "approve"
    print("✅ final_recommendation normalized to recommended_action")
except Exception as e:
    print(f"❌ Normalization failed: {e}")
    sys.exit(1)

# Test 6: Check project files
print("\n[TEST 6] Verifying project structure...")
required_files = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    ".gitignore",
    "loan_evaluator/__init__.py",
    "loan_evaluator/loan_evaluator.py",
    "loan_evaluator/models.py",
    "loan_evaluator/trace_graph.py",
    "loan_evaluator/evaluator_v1.ipynb",
]

for file_path in required_files:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path}: NOT FOUND")
        sys.exit(1)

# Test 7: Verify LangSmith tracing configuration guard
print("\n[TEST 7] Checking LangSmith tracing configuration...")
try:
    import os
    from unittest.mock import patch

    from loan_evaluator.loan_evaluator import LoanEvaluator

    with patch.dict(os.environ, {}, clear=True):
        try:
            LoanEvaluator(use_langsmith=True)
            raise AssertionError("Expected missing LANGSMITH_API_KEY error")
        except RuntimeError as e:
            assert "LANGSMITH_API_KEY" in str(e)

    print("✅ Missing LangSmith API key fails fast")
except Exception as e:
    print(f"❌ LangSmith configuration check failed: {e}")
    sys.exit(1)

# Test 8: Verify redacted LangSmith trace payloads
print("\n[TEST 8] Checking redacted LangSmith trace payloads...")
try:
    from loan_evaluator.trace_graph import (
        agent_output_trace_payload,
        application_trace_summary,
        rationale_trace_summary,
    )
    from langgraph.graph import StateGraph

    trace_input = application_trace_summary(app)
    assert "applicant_name" not in trace_input
    assert "annual_income" not in trace_input
    assert "loan_purpose" in trace_input

    trace_agent_output = agent_output_trace_payload(review.model_dump_json())
    assert trace_agent_output["raw_output_redacted"] is True
    assert "raw_output" not in trace_agent_output
    assert trace_agent_output["summary"]["recommended_action"] == review.recommended_action

    rationale_summary = rationale_trace_summary([review])
    assert rationale_summary["rationale_redacted"] is True
    assert "review_reasoning" not in rationale_summary
    assert rationale_summary["agent_count"] == 1
    assert rationale_summary["actions"][review.recommended_action] == 1
    assert rationale_summary["review_signals"][0]["strength_count"] == 2

    rationale_with_text = rationale_trace_summary(
        [review],
        include_reasoning_text=True,
    )
    assert rationale_with_text["rationale_redacted"] is False
    assert rationale_with_text["review_reasoning"][0]["strengths"] == review.strengths
    assert rationale_with_text["review_reasoning"][0]["risks"] == review.risks

    assert StateGraph is not None
    print("✅ Trace payloads are redacted and LangGraph is available")
except Exception as e:
    print(f"❌ Trace payload check failed: {e}")
    sys.exit(1)

# Test 9: Test EvaluationResult creation
print("\n[TEST 9] Creating evaluation result object...")
try:
    with open(project_root / "loan_evaluator" / "sample_data" / "loan_application_1.json") as f:
        app_data = json.load(f)
    
    app = LoanApplication(**app_data)
    
    result = EvaluationResult(
        application_id=app.applicant_name,
        applicant_name=app.applicant_name,
        submission_date=app.submission_date,
        reviews=[review],
        final_recommendation="approve",
        overall_score=75.5,
        decision_rationale="Strong profile with stable income",
        avg_confidence=0.85,
        agents_involved=["credit_risk_analyst"],
    )
    
    print(f"✅ Evaluation result created:")
    print(f"   - Applicant: {result.applicant_name}")
    print(f"   - Recommendation: {result.final_recommendation}")
    print(f"   - Score: {result.overall_score}/100")
except Exception as e:
    print(f"❌ Evaluation result creation failed: {e}")
    sys.exit(1)

# Test 10: JSON serialization
print("\n[TEST 10] Testing JSON serialization...")
try:
    json_str = result.model_dump_json(indent=2)
    json_data = json.loads(json_str)
    print(f"✅ Serialization successful: {len(json_str)} bytes")
except Exception as e:
    print(f"❌ Serialization failed: {e}")
    sys.exit(1)

# Test 11: Calculate metrics
print("\n[TEST 11] Testing financial calculations...")
try:
    dti = (app.existing_debt * 12 / app.annual_income) if app.annual_income else 0
    ltv = (app.requested_amount / app.collateral_value * 100) if app.collateral_value else 0
    
    print(f"✅ Debt-to-Income Ratio: {dti:.2%}")
    print(f"✅ Loan-to-Value Ratio: {ltv:.1f}%")
except Exception as e:
    print(f"❌ Calculation failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print(f"\nProject Status:")
print(f"  • Data models: ✅ Working")
print(f"  • Sample data: ✅ {len(sample_files)} applications loaded")
print(f"  • Prompts: ✅ {len(expected_prompts)} templates present")
print(f"  • JSON serialization: ✅ Working")
print(f"  • Financial calculations: ✅ Working")
print(f"\nNote: Full agent evaluation requires AWS Bedrock and 'strands-agents'.")
print(f"Install with: pip install -r requirements.txt")
print("\n" + "=" * 70)
