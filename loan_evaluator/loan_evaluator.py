"""
Loan Application Evaluator using Strands Agents and AWS Bedrock.

This module implements a multi-agent evaluation system for loan applications
using specialized agents for credit analysis, compliance review, fraud detection,
and final loan officer recommendation.
"""

from __future__ import annotations

import json
import asyncio
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from strands import Agent
from strands.models import BedrockModel

DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

BEDROCK_MODEL_ALIASES = {
    # Anthropic API names are not valid Bedrock model identifiers.
    "claude-3-5-sonnet-20241022": DEFAULT_BEDROCK_MODEL_ID,
    "anthropic.claude-3-5-sonnet-20241022": DEFAULT_BEDROCK_MODEL_ID,
    # Claude 3.5 Sonnet v2 is commonly invoked through the US inference profile.
    "anthropic.claude-3-5-sonnet-20241022-v2:0": DEFAULT_BEDROCK_MODEL_ID,
}

try:
    from .models import (
        LoanApplication,
        ReviewResult,
        EvaluationResult,
        CreditAnalysisDetail,
        FraudAnalysisDetail,
        ComplianceAnalysisDetail,
    )
    from .trace_graph import (
        agent_output_trace_payload,
        run_langsmith_trace_graph,
    )
except ImportError:
    from models import (
        LoanApplication,
        ReviewResult,
        EvaluationResult,
        CreditAnalysisDetail,
        FraudAnalysisDetail,
        ComplianceAnalysisDetail,
    )
    from trace_graph import (
        agent_output_trace_payload,
        run_langsmith_trace_graph,
    )


class LoanEvaluator:
    """Multi-agent loan evaluation orchestrator."""

    def __init__(
        self,
        model_name: str = DEFAULT_BEDROCK_MODEL_ID,
        region: str = "us-east-1",
        use_langsmith: bool = False,
        include_reasoning_in_traces: bool = False,
        use_langfuse: Optional[bool] = None,
    ):
        """
        Initialize the loan evaluator.

        Args:
            model_name: AWS Bedrock model ID
            region: AWS region
            use_langsmith: Enable LangSmith tracing
            include_reasoning_in_traces: Include review text in LangSmith traces
            use_langfuse: Deprecated alias for use_langsmith
        """
        model_id = self._normalize_bedrock_model_id(model_name)
        self.model = BedrockModel(
            model_id=model_id,
            region_name=region,
        )
        if use_langfuse is not None:
            use_langsmith = use_langfuse

        self.use_langsmith = use_langsmith
        self.include_reasoning_in_traces = include_reasoning_in_traces
        self.langsmith_project = os.getenv("LANGSMITH_PROJECT", "strands-agents-loan")
        self.langsmith_client = None
        if self.use_langsmith:
            self.langsmith_client = self._initialize_langsmith_client()

        self.prompts = self._load_prompts()
        self.evaluators = self._initialize_agents()

    @staticmethod
    def _normalize_bedrock_model_id(model_name: str) -> str:
        """Convert common Claude API model names to Bedrock model/profile IDs."""
        return BEDROCK_MODEL_ALIASES.get(model_name, model_name)

    def _initialize_langsmith_client(self):
        """Create a LangSmith client and fail fast on missing tracing config."""
        if not os.getenv("LANGSMITH_API_KEY"):
            raise RuntimeError(
                "LANGSMITH_API_KEY is required when use_langsmith=True."
            )

        os.environ.setdefault("LANGSMITH_TRACING", "true")

        from langsmith import Client

        return Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_ENDPOINT"),
        )

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates for each agent."""
        prompts_dir = Path(__file__).parent / "prompts"
        prompts = {}

        prompt_files = {
            "credit_analyst": "credit_analyst_prompt.txt",
            "compliance_officer": "compliance_prompt.txt",
            "fraud_detector": "fraud_detection_prompt.txt",
            "loan_officer": "loan_officer_prompt.txt",
        }

        for key, filename in prompt_files.items():
            filepath = prompts_dir / filename
            if filepath.exists():
                with open(filepath, "r") as f:
                    prompts[key] = f.read()
            else:
                print(f"Warning: Prompt file not found: {filepath}")
                prompts[key] = f"You are a {key}. Evaluate the loan application."

        return prompts

    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize specialized evaluation agents."""
        agents = {}

        # Credit Risk Analyst
        agents["credit_analyst"] = Agent(
            name="credit_analyst",
            system_prompt=self.prompts.get("credit_analyst", ""),
            model=self.model,
            callback_handler=None,
        )

        # Compliance Officer
        agents["compliance_officer"] = Agent(
            name="compliance_officer",
            system_prompt=self.prompts.get("compliance_officer", ""),
            model=self.model,
            callback_handler=None,
        )

        # Fraud Detection Specialist
        agents["fraud_detector"] = Agent(
            name="fraud_detector",
            system_prompt=self.prompts.get("fraud_detector", ""),
            model=self.model,
            callback_handler=None,
        )

        # Loan Officer (Final Decision)
        agents["loan_officer"] = Agent(
            name="loan_officer",
            system_prompt=self.prompts.get("loan_officer", ""),
            model=self.model,
            callback_handler=None,
        )

        return agents

    def _format_application_for_review(
        self, application: LoanApplication
    ) -> str:
        """Format application data as readable text for agents."""
        debt_to_income = (
            (application.existing_debt * 12 / application.annual_income)
            if application.annual_income
            else 0
        )

        app_text = f"""
LOAN APPLICATION REVIEW
======================
Application ID: {application.applicant_name}
Submission Date: {application.submission_date}

APPLICANT INFORMATION
---------------------
Name: {application.applicant_name}
Email: {application.applicant_email}
Phone: {application.applicant_phone}
Date of Birth: {application.date_of_birth}
Current Address: {application.address} (for {application.years_at_address} years)

LOAN DETAILS
------------
Requested Amount: ${application.requested_amount:,.2f}
Purpose: {application.loan_purpose}
Term: {application.loan_term_months} months
Down Payment: ${application.down_payment or 0:,.2f}
Collateral Value: ${application.collateral_value or 0:,.2f}

EMPLOYMENT & INCOME
-------------------
Status: {application.employment_status}
Employer: {application.current_employer}
Years Employed: {application.employment_years}
Annual Income: ${application.annual_income:,.2f}
Secondary Income: ${application.secondary_income or 0:,.2f}
Income Source: {application.income_source}

CREDIT PROFILE
--------------
Credit Score: {application.credit_score}
Recent Inquiries (6 months): {application.credit_inquiries_6_months}
Existing Monthly Debt: ${application.existing_debt:,.2f}
Debt-to-Income Ratio: {debt_to_income:.2%}
Bankruptcy History: {application.bankruptcy_history or "None reported"}

FINANCIAL POSITION
------------------
Liquid Assets: ${application.liquid_assets:,.2f}
Loan-to-Value: {(application.requested_amount / application.collateral_value * 100 if application.collateral_value else 0):.1f}%

DOCUMENTATION
--------------
Documents Provided: {", ".join(application.documents_provided)}

APPLICANT NOTES
---------------
{application.notes or "No additional notes"}
"""
        return app_text

    async def _get_agent_review(
        self, agent: Agent, application_text: str, reviewer_type: str
    ) -> ReviewResult:
        """Get evaluation from a single agent."""
        prompt = f"""
Review this loan application and provide your assessment.

{application_text}

Remember to return ONLY valid JSON matching the specified format for {reviewer_type}.
The JSON object must include "recommended_action" with one of:
"approve", "conditional_approval", "deny", "request_more_info", or "manual_review".
"""

        response = await self._invoke_agent(agent, prompt, reviewer_type)

        # Parse JSON response
        try:
            # Extract JSON from response
            json_str = str(response)
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            json_data = json.loads(json_str)
            json_data = self._normalize_review_json(json_data, reviewer_type)
            return ReviewResult(**json_data)

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback if parsing fails
            print(f"Warning: Could not parse {reviewer_type} response: {e}")
            return ReviewResult(
                reviewer_name=reviewer_type,
                score=50,
                strengths=["Unable to parse response"],
                weaknesses=[],
                risks=[],
                recommended_action="request_more_info",
                confidence=0.1,
            )

    @staticmethod
    def _normalize_review_json(
        json_data: Dict[str, Any], reviewer_type: str
    ) -> Dict[str, Any]:
        """Normalize common LLM response variants into ReviewResult fields."""
        normalized = dict(json_data)

        if "recommended_action" not in normalized:
            for alias in ("final_recommendation", "recommendation", "action"):
                if alias in normalized:
                    normalized["recommended_action"] = normalized[alias]
                    break

        if reviewer_type == "loan_officer":
            normalized["reviewer_name"] = "loan_officer"

        return normalized

    async def _invoke_agent(
        self, agent: Agent, prompt: str, reviewer_type: str
    ) -> object:
        """Invoke an agent, optionally wrapping the call in a LangSmith LLM run."""
        if not self.use_langsmith:
            return await agent.invoke_async(prompt)

        from langsmith import traceable

        model_id = self.model.config.get("model_id")

        def process_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "reviewer_type": reviewer_type,
                "prompt_redacted": True,
                "prompt_length": len(inputs["prompt_text"]),
            }

        def process_outputs(output: object) -> Dict[str, Any]:
            return agent_output_trace_payload(output)

        @traceable(
            run_type="llm",
            name=f"{reviewer_type}-review",
            metadata={
                "reviewer_type": reviewer_type,
                "agent_name": getattr(agent, "name", None),
                "ls_provider": "aws-bedrock",
                "ls_model_name": model_id,
            },
            process_inputs=process_inputs,
            process_outputs=process_outputs,
        )
        async def invoke_with_trace(prompt_text: str) -> object:
            return await agent.invoke_async(prompt_text)

        return await invoke_with_trace(prompt)

    def evaluate(self, application: LoanApplication) -> EvaluationResult:
        """
        Perform complete multi-agent evaluation of a loan application.

        Args:
            application: The loan application to evaluate

        Returns:
            Complete evaluation result with all reviews and final recommendation
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_async(application))

        raise RuntimeError(
            "LoanEvaluator.evaluate() cannot run inside an active event loop. "
            "Use 'await evaluator.evaluate_async(application)' in notebooks."
        )

    async def evaluate_async(self, application: LoanApplication) -> EvaluationResult:
        """
        Perform complete multi-agent evaluation of a loan application asynchronously.

        Use this method from notebooks and other async environments.
        """
        if self.use_langsmith:
            return await run_langsmith_trace_graph(self, application)

        return await self._evaluate_async_impl(application)

    async def _evaluate_async_impl(
        self, application: LoanApplication
    ) -> EvaluationResult:
        """Run the evaluator inside the active tracing context."""
        # Format application
        app_text = self._format_application_for_review(application)

        # Run all agents in sequence (or parallel in async context)
        reviews = []

        # Credit Analyst
        print("Running Credit Risk Analysis...")
        credit_review = await self._get_agent_review(
            self.evaluators["credit_analyst"],
            app_text,
            "credit_risk_analyst",
        )
        reviews.append(credit_review)

        # Compliance Officer
        print("Running Compliance Review...")
        compliance_review = await self._get_agent_review(
            self.evaluators["compliance_officer"],
            app_text,
            "compliance_officer",
        )
        reviews.append(compliance_review)

        # Fraud Detector
        print("Running Fraud Detection...")
        fraud_review = await self._get_agent_review(
            self.evaluators["fraud_detector"],
            app_text,
            "fraud_detection_specialist",
        )
        reviews.append(fraud_review)

        # Generate synthesis prompt for Loan Officer
        synthesis_prompt = self._create_synthesis_prompt(application, reviews)

        # Loan Officer (Final Decision)
        print("Getting Final Loan Officer Review...")
        loan_officer_review = await self._get_agent_review(
            self.evaluators["loan_officer"],
            synthesis_prompt,
            "loan_officer",
        )
        reviews.append(loan_officer_review)

        # Create final evaluation result
        evaluation = self._synthesize_results(
            application, reviews, loan_officer_review
        )

        return evaluation

    def _flush_langsmith(self) -> None:
        """Flush LangSmith background tracing when the installed SDK supports it."""
        if self.langsmith_client and hasattr(self.langsmith_client, "flush"):
            self.langsmith_client.flush()

    def _run_agent_sync(
        self, agent: Agent, prompt_text: str, reviewer_type: str
    ) -> ReviewResult:
        """Synchronous wrapper for agent execution."""
        return asyncio.run(self._get_agent_review(agent, prompt_text, reviewer_type))

    def _create_synthesis_prompt(
        self, application: LoanApplication, reviews: List[ReviewResult]
    ) -> str:
        """Create synthesis prompt combining all reviews for final decision."""
        app_text = self._format_application_for_review(application)

        reviews_summary = "SPECIALIST REVIEWS:\n" + "=" * 50 + "\n"
        for review in reviews:
            reviews_summary += f"""
{review.reviewer_name.upper()}
Score: {review.score}/100
Recommendation: {review.recommended_action}
Confidence: {review.confidence:.1%}
Strengths: {", ".join(review.strengths[:2]) if review.strengths else "None"}
Key Risks: {", ".join(review.risks[:2]) if review.risks else "None"}
"""

        return f"""
{app_text}

{reviews_summary}

As the Loan Officer, synthesize all specialist reviews above and provide your 
final recommendation. Consider all perspectives and provide a holistic assessment.
Return ONLY strict JSON with "recommended_action", not "final_recommendation".
"""

    def _synthesize_results(
        self,
        application: LoanApplication,
        reviews: List[ReviewResult],
        final_review: ReviewResult,
    ) -> EvaluationResult:
        """Synthesize all reviews into final evaluation result."""
        # Calculate weighted scores
        weights = {
            "credit_risk_analyst": 0.35,
            "compliance_officer": 0.25,
            "fraud_detection_specialist": 0.25,
            "loan_officer": 0.15,  # Meta reviewer
        }

        overall_score = sum(
            r.score * weights.get(r.reviewer_name, 0.25) for r in reviews
        )

        # Determine final recommendation
        final_recommendation = final_review.recommended_action
        if final_recommendation == "conditional_approval":
            final_recommendation = "conditional_approval"

        # Extract recommended terms
        recommended_terms = {}
        for review in reviews:
            if review.recommended_terms:
                recommended_terms.update(review.recommended_terms)

        # Build decision rationale
        rationale_parts = []
        if final_review.overall_rationale:
            rationale_parts.append(final_review.overall_rationale)
        else:
            for review in reviews[:3]:  # Summarize first 3 reviews
                if review.strengths:
                    rationale_parts.append(
                        f"Strengths: {', '.join(review.strengths[:1])}"
                    )

        return EvaluationResult(
            application_id=application.applicant_name,
            applicant_name=application.applicant_name,
            submission_date=application.submission_date,
            reviews=reviews,
            final_recommendation=final_recommendation,
            overall_score=overall_score,
            decision_rationale=" | ".join(rationale_parts),
            avg_confidence=sum(r.confidence for r in reviews) / len(reviews),
            recommended_interest_rate=recommended_terms.get("interest_rate"),
            recommended_loan_amount=recommended_terms.get("loan_amount"),
            special_conditions=recommended_terms.get("special_conditions", []),
            agents_involved=[r.reviewer_name for r in reviews],
        )

    def export_result(
        self, result: EvaluationResult, filepath: str = None
    ) -> str:
        """Export evaluation result to JSON file."""
        if filepath is None:
            filepath = f"evaluation_{result.application_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filepath, "w") as f:
            f.write(result.model_dump_json(indent=2))

        print(f"Evaluation exported to: {filepath}")
        return filepath


# Example usage
if __name__ == "__main__":
    # Load sample application
    sample_data_path = Path(__file__).parent / "sample_data" / "loan_application_1.json"

    with open(sample_data_path, "r") as f:
        app_data = json.load(f)

    application = LoanApplication(**app_data)

    # Create evaluator
    evaluator = LoanEvaluator()

    # Run evaluation
    result = evaluator.evaluate(application)

    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Applicant: {result.applicant_name}")
    print(f"Final Recommendation: {result.final_recommendation}")
    print(f"Overall Score: {result.overall_score:.1f}/100")
    print(f"Average Confidence: {result.avg_confidence:.1%}")

    print("\nIndividual Reviews:")
    for review in result.reviews:
        print(f"  - {review.reviewer_name}: {review.score}/100 ({review.recommended_action})")

    # Export results
    evaluator.export_result(result)
