"""
core/learning/learning_engine.py
----------------------------------
Layer 10: Learning Engine

Maps feedback classification → parameter adjustment.
Writes changes directly to the global CONFIG singleton so every
subsequent query inherits the improvement.

Parameter adjustment rules:
  retrieval_error  → top_k ↑  (we missed relevant docs)
  ranking_error    → threshold ↓  (we filtered out good chunks)
  hallucination    → confidence threshold ↑  (we accepted bad answers)
  vague_query      → log only (query preprocessing issue, not retrieval)
  reasoning_error  → log only (LLM reasoning issue, not config)
  correct          → no change
  irrelevant_answer→ top_k ↑ + threshold ↓ (broad failure)

All adjustments are bounded to prevent runaway drift.
"""

import json
import os
from datetime import datetime

from core.learning.feedback_classifier import FeedbackClassification
from app.config import CONFIG
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)

LEARNING_LOG = "logs/learning_updates.jsonl"


class LearningEngine:
    """
    Adjusts CONFIG parameters based on classified feedback.
    Persists each learning event to a JSONL log for audit and analysis.
    """

    # Bounds for parameter values
    _TOP_K_MIN, _TOP_K_MAX       = 5,   60
    _THRESHOLD_MIN, _MAX         = 2.0, 9.0
    _CONFIDENCE_MIN, _CONF_MAX   = 0.50, 0.95

    def update(
        self,
        classification: FeedbackClassification,
        query_id: str = "",
    ) -> dict:
        """
        Apply parameter update based on feedback category.
        Returns a dict of changes made (for logging and testing).
        """
        category = classification.category
        changes:  dict = {"category": category, "query_id": query_id}

        with log_stage(logger, "learning_update", query_id=query_id) as ctx:
            if category == "retrieval_error":
                # We missed relevant docs → cast a wider net
                old = CONFIG.retrieval.top_k_semantic
                CONFIG.retrieval.top_k_semantic = min(old + 3, self._TOP_K_MAX)
                CONFIG.retrieval.top_k_keyword  = min(
                    CONFIG.retrieval.top_k_keyword + 3, self._TOP_K_MAX
                )
                changes["top_k_semantic"] = f"{old} → {CONFIG.retrieval.top_k_semantic}"

            elif category == "ranking_error":
                # Good chunks filtered out → loosen the threshold
                old = CONFIG.ranking.threshold
                CONFIG.ranking.threshold = max(old - 0.5, self._THRESHOLD_MIN)
                changes["ranking_threshold"] = f"{old} → {CONFIG.ranking.threshold}"

            elif category == "hallucination":
                # We accepted an ungrounded answer → raise the confidence bar
                old = CONFIG.confidence.min_confidence
                CONFIG.confidence.min_confidence = min(old + 0.05, self._CONF_MAX)
                changes["min_confidence"] = f"{old} → {CONFIG.confidence.min_confidence}"

            elif category == "irrelevant_answer":
                # Broad failure — loosen both retrieval and ranking
                CONFIG.retrieval.top_k_semantic = min(
                    CONFIG.retrieval.top_k_semantic + 2, self._TOP_K_MAX
                )
                CONFIG.ranking.threshold = max(
                    CONFIG.ranking.threshold - 0.25, self._THRESHOLD_MIN
                )
                changes["broad_adjustment"] = "top_k+2, threshold-0.25"

            elif category == "correct":
                # Optionally tighten params slightly over time to stay efficient
                changes["action"] = "no_change"

            else:
                # vague_query, reasoning_error — config won't fix these
                changes["action"] = "log_only"

            ctx["changes"] = str(changes)

        self._persist(changes)
        return changes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, changes: dict) -> None:
        """Append learning event to JSONL audit log."""
        os.makedirs(os.path.dirname(LEARNING_LOG), exist_ok=True)
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "config_snapshot": {
                "top_k_semantic":   CONFIG.retrieval.top_k_semantic,
                "top_k_keyword":    CONFIG.retrieval.top_k_keyword,
                "ranking_threshold":CONFIG.ranking.threshold,
                "min_confidence":   CONFIG.confidence.min_confidence,
            },
            **changes,
        }
        with open(LEARNING_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
