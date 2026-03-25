"""
core/evaluation/confidence_evaluator.py
-----------------------------------------
Layer 7: Confidence Evaluation

A SEPARATE 70B call acts as grounding judge.
This is intentionally separate from the generator — asking the
same model to judge its own output is a conflict of interest.

Output: ConfidenceResult (Pydantic-validated)
"""

from typing import List

from pydantic import BaseModel, Field

from llm.groq_client import groq_client
from llm.prompts import confidence_evaluator_prompt
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)


class ConfidenceResult(BaseModel):
    confidence:        float       = Field(ge=0.0, le=1.0)
    grounded:          bool
    unsupported_claims: List[str]  = []
    reasoning:         str         = ""


class ConfidenceEvaluator:
    """
    Evaluates whether an answer is grounded in the provided context.
    Returns a structured ConfidenceResult.
    """

    def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: List[str],
        query_id: str = "",
    ) -> ConfidenceResult:

        context = "\n\n".join(
            f"[{i+1}] {c}" for i, c in enumerate(context_chunks)
        )

        with log_stage(logger, "confidence_evaluation", query_id=query_id) as ctx:
            messages = confidence_evaluator_prompt(query, answer, context)
            result: ConfidenceResult = groq_client.call_70b(
                messages,
                json_mode=True,
                schema=ConfidenceResult,
                max_tokens=256,
            )
            ctx["confidence"] = result.confidence
            ctx["grounded"]   = result.grounded
            ctx["unsupported_claims"] = len(result.unsupported_claims)
            
        return result
            
    def _context_precision(self, chunks, ground_truth):
            if not chunks:
                return 0.0

            relevant = sum(
                1 for c in chunks if ground_truth.lower() in c.lower()
            )
            return relevant / len(chunks)
        
    def _context_recall(self, chunks, ground_truth):
        if not ground_truth:
            return 0.0

        combined = " ".join(chunks).lower()
        return 1.0 if ground_truth.lower() in combined else 0.0

