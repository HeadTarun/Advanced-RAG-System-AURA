"""
core/learning/feedback_classifier.py
--------------------------------------
Layer 9: Feedback Classification

Uses LLaMA 3 8B to classify user feedback into one of 7 error categories.
This is cheap (8B) because it's a classification task, not reasoning.

Categories:
  correct | retrieval_error | ranking_error | reasoning_error |
  hallucination | vague_query | irrelevant_answer
"""

from pydantic import BaseModel, Field

from llm.groq_client import groq_client
from llm.prompts import feedback_classifier_prompt
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)

VALID_CATEGORIES = {
    "correct",
    "retrieval_error",
    "ranking_error",
    "reasoning_error",
    "hallucination",
    "vague_query",
    "irrelevant_answer",
}


class FeedbackClassification(BaseModel):
    category:    str   = Field(description="One of the 7 error categories")
    confidence:  float = Field(ge=0.0, le=1.0)
    explanation: str   = ""

    def model_post_init(self, __context):
        if self.category not in VALID_CATEGORIES:
            self.category = "irrelevant_answer"   # safe default


class FeedbackClassifier:
    """Classifies user feedback into a structured error category."""

    def classify(
        self,
        query:    str,
        answer:   str,
        feedback: str,
        query_id: str = "",
    ) -> FeedbackClassification:

        with log_stage(logger, "feedback_classification", query_id=query_id) as ctx:
            messages = feedback_classifier_prompt(query, answer, feedback)
            result: FeedbackClassification = groq_client.call_8b(
                messages,
                json_mode=True,
                schema=FeedbackClassification,
                max_tokens=256,
            )
            ctx["category"]   = result.category
            ctx["confidence"] = result.confidence

        return result
