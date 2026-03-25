"""
core/generation/answer_generator.py
-------------------------------------
Layer 6: Answer Generation

Uses LLaMA 3 70B.
Strictly grounded — every claim must cite a chunk index.
The prompt forbids external knowledge injection.
"""

from typing import List

from llm.groq_client import groq_client
from llm.prompts import answer_generator_prompt
from app.config import CONFIG
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)


class AnswerGenerator:
    """Generates grounded answers from context chunks."""

    def generate(
        self,
        query: str,
        context_chunks: List[str],
        memory_context: str = "",
        query_id: str = "",
    ) -> str:
        """
        Formats context as a numbered list so the model can cite chunk [N],
        then calls the 70B model for generation.
        """
        if not context_chunks:
            return "I cannot answer this — no relevant context was retrieved."

        # Number the chunks so the model can cite them
        numbered = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        )

        with log_stage(logger, "answer_generation", query_id=query_id) as ctx:
            messages = answer_generator_prompt(query, numbered, memory_context)
            answer = groq_client.call_70b(
                messages,
                json_mode=False,
                max_tokens=CONFIG.tokens.max_answer_tokens,
            )
            ctx["answer_tokens"] = len(answer.split())

        return answer.strip()
