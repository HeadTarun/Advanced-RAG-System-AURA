"""
core/optimizer/context_optimizer.py
-------------------------------------
Layer 4: Context Optimization

FIX APPLIED: Query-aware summarization (reviewer gap #3).
Each chunk is summarised relative to the specific query,
not generically — this produces tighter, more relevant context.

Steps:
  1. Deduplicate chunks by embedding cosine similarity
  2. Query-aware summarise each surviving chunk (70B)
  3. Enforce token budget — drop last chunks if over limit
"""

from typing import List

from core.ranking.ranker import RankedChunk
from core.retriever.faiss_retriever import FaissRetriever
from llm.groq_client import groq_client
from llm.prompts import summarizer_prompt
from app.config import CONFIG
from utils.helpers import deduplicate_chunks, truncate_to_budget, count_tokens
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)


class ContextOptimizer:
    """
    Deduplicates → query-aware summarises → enforces token budget.
    """

    def __init__(self, retriever: FaissRetriever) -> None:
        self._retriever = retriever  # used for embedding during dedup

    def optimize(
        self,
        query: str,
        ranked_chunks: List[RankedChunk],
        query_id: str = "",
    ) -> List[str]:
        """
        Returns a list of optimised context strings ready for the generator.
        """
        if not ranked_chunks:
            return []

        budget    = CONFIG.tokens.max_context_tokens
        texts     = [rc.text for rc in ranked_chunks]

        with log_stage(logger, "context_optimization", query_id=query_id) as ctx:
            # Step 1: Deduplicate
            embeddings = self._retriever.embed_batch(texts)
            unique_texts, _ = deduplicate_chunks(texts, embeddings, threshold=0.92)
            ctx["after_dedup"] = len(unique_texts)

            # Step 2: Query-aware summarisation
            summarised = []
            for chunk in unique_texts:
                if count_tokens(chunk) > 200:
                    # Only summarise long chunks — short ones are fine as-is
                    summary = groq_client.call_70b(
                        summarizer_prompt(query, chunk),
                        json_mode=False,
                        max_tokens=300,
                    )
                    summarised.append(summary.strip())
                else:
                    summarised.append(chunk)

            # Step 3: Token budget enforcement
            final = truncate_to_budget(summarised, budget)
            ctx["after_budget"] = len(final)
            ctx["total_tokens"] = sum(count_tokens(t) for t in final)

        return final
