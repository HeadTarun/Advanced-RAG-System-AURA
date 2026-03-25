"""
core/failure/failure_handler.py
---------------------------------
Layer 8: Failure Handling

When confidence < threshold:
  - Attempt 1: Retry with relaxed retrieval (higher top_k, lower threshold)
  - Attempt 2: Retry with even looser params
  - Final: Return safe fallback message

The retry modifies CONFIG in-place so the next real query
inherits improved parameters if the failure was systematic.
"""

from typing import Callable, Optional

from core.evaluation.confidence_evaluator import ConfidenceResult
from app.config import CONFIG
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)

FALLBACK_MESSAGE = (
    "I was unable to generate a confident, grounded answer from the "
    "available context. Please try rephrasing your question or providing "
    "more specific context."
)


class FailureHandler:
    """
    Manages retry logic when confidence evaluation fails.

    Usage:
        result, answer = handler.handle(
            confidence_result,
            attempt=0,
            retry_fn=lambda: pipeline._run_retrieval_to_generation(query),
        )
    """

    def handle(
        self,
        confidence_result: ConfidenceResult,
        attempt: int,
        retry_fn: Optional[Callable] = None,
        query_id: str = "",
    ) -> tuple[bool, str]:
        """
        Returns (should_retry, message).
        If should_retry=True, caller should call retry_fn and re-evaluate.
        """
        max_retries = CONFIG.confidence.max_retries
        threshold   = CONFIG.confidence.min_confidence

        if confidence_result.confidence >= threshold:
            return False, ""   # No failure

        with log_stage(logger, "failure_handling", query_id=query_id) as ctx:
            ctx["confidence"]  = confidence_result.confidence
            ctx["attempt"]     = attempt
            ctx["max_retries"] = max_retries

            if attempt >= max_retries:
                ctx["action"] = "fallback"
                return False, FALLBACK_MESSAGE

            # Relax retrieval parameters for retry
            CONFIG.retrieval.top_k_semantic     = min(
                CONFIG.retrieval.top_k_semantic + 5, 50
            )
            CONFIG.retrieval.top_k_keyword      = min(
                CONFIG.retrieval.top_k_keyword + 5, 50
            )
            CONFIG.ranking.threshold            = max(
                CONFIG.ranking.threshold - 0.5, 3.0
            )
            ctx["action"]        = "retry"
            ctx["new_top_k"]     = CONFIG.retrieval.top_k_semantic
            ctx["new_threshold"] = CONFIG.ranking.threshold

        return True, ""   # Caller should retry
