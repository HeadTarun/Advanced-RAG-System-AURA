"""
core/ranking/ranker.py
----------------------
Layer 3: Ranking & Filtering

FIXES APPLIED:

  Bug 1 — Hard threshold eliminates ALL chunks when corpus is small / focused.
           CONFIG.ranking.threshold = 6.0 is calibrated for a large noisy
           corpus. With 21–100 tightly topical chunks the LLM legitimately
           scores everything 4–5 (correct relative to its rubric) and the
           hard filter wipes them all out. Result: after_llm_filter = 0 on
           every query even though retrieval is working perfectly.

           Fix A — Adaptive threshold: if fewer than `min_results` chunks
           survive the hard threshold, fall back to the top-N by raw LLM
           score regardless of threshold. The pipeline always has something
           to work with; low-confidence answers are surfaced rather than
           silently swallowed.

           Fix B — Score ALL stage-1 chunks before filtering. The previous
           code appended to ranked_chunks inside the loop and applied the
           threshold immediately, so a single low-scoring chunk early in
           the loop could hide higher-scoring ones that came later. All
           chunks are now scored first, then filtered / sorted once.

  Bug 2 — max_chunks_to_llm cap was applied AFTER the threshold filter,
           so with a strict threshold it was cutting an already-empty list.
           It is now applied to the final sorted list, which is correct.

  No other logic changed.
"""

from typing import List, Tuple

from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from llm.groq_client import groq_client
from llm.prompts import ranker_prompt
from app.config import CONFIG
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Minimum chunks that must survive to constitute a "real" result set.
# If the hard threshold produces fewer than this, we fall back to top-N
# by score (ignoring the threshold) so the pipeline never starves.
MIN_RESULTS_BEFORE_FALLBACK = 2


class ChunkScore(BaseModel):
    score:  int   # 1–10
    reason: str


class RankedChunk:
    def __init__(self, text: str, cross_score: float, score: int, reason: str):
        self.text        = text
        self.cross_score = cross_score
        self.score   = score
        self.reason      = reason

    def __repr__(self) -> str:
        return f"RankedChunk(llm={self.score}, cross={self.cross_score:.3f})"


class Ranker:
    """
    Two-stage ranker.  Dependency-injected cross-encoder and Groq client.
    """

    def __init__(self) -> None:
        self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("ranker_ready", cross_encoder=CROSS_ENCODER_MODEL)

    # ------------------------------------------------------------------
    # Stage 1: Cross-encoder
    # ------------------------------------------------------------------

    def _cross_encode(
        self, query: str, chunks: List[str], top_k: int
    ) -> List[Tuple[str, float]]:
        """Score all chunks with the cross-encoder, return top_k."""
        pairs  = [[query, chunk] for chunk in chunks]
        scores = self._cross_encoder.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Stage 2: LLM scoring
    # ------------------------------------------------------------------

    def _llm_score(self, query: str, chunk: str) -> ChunkScore:
        messages = ranker_prompt(query, chunk)
        return groq_client.call_70b(
            messages, json_mode=True, schema=ChunkScore, max_tokens=128
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        query_id: str = "",
    ) -> List[RankedChunk]:
        """
        Returns sorted, filtered list of RankedChunk objects.

        Guarantee: if ANY candidates exist, at least MIN_RESULTS_BEFORE_FALLBACK
        chunks are returned (unless the entire candidate list is smaller than
        that constant, in which case all survivors are returned).
        """
        chunks       = [c for c, _ in candidates]
        threshold    = CONFIG.ranking.threshold
        top_k_rerank = CONFIG.retrieval.top_k_rerank
        max_to_llm   = CONFIG.ranking.max_chunks_to_llm

        with log_stage(logger, "ranking", query_id=query_id) as ctx:

            # Stage 1 — cross-encoder narrows the candidate set
            stage1 = self._cross_encode(query, chunks, top_k=top_k_rerank)
            ctx["after_cross_encoder"] = len(stage1)

            # Stage 2 — score ALL stage-1 chunks with the LLM first,
            # then filter. Scoring inside the filter loop (old behaviour)
            # caused early low scorers to shadow later high scorers.
            all_scored: List[RankedChunk] = []
            for chunk_text, cross_score in stage1:
                cs = self._llm_score(query, chunk_text)
                all_scored.append(
                    RankedChunk(chunk_text, cross_score, cs.score, cs.reason)
                )

            # Sort by LLM score desc, then cross score desc
            all_scored.sort(
                key=lambda x: (x.score, x.cross_score), reverse=True
            )

            # Apply hard threshold
            above_threshold = [c for c in all_scored if cs.score >= threshold]

            # FIX: adaptive fallback — if too few chunks survive the hard
            # threshold, use the top-N by score so the pipeline never
            # returns empty when good chunks exist.
            if len(above_threshold) >= MIN_RESULTS_BEFORE_FALLBACK:
                final = above_threshold[:max_to_llm]
                ctx["threshold_applied"] = threshold
            else:
                fallback_n = max(MIN_RESULTS_BEFORE_FALLBACK, max_to_llm)
                final = all_scored[:fallback_n]
                ctx["threshold_applied"] = "fallback"
                logger.warning(
                    "ranker_threshold_fallback",
                    query_id=query_id,
                    above_threshold=len(above_threshold),
                    fallback_n=len(final),
                    configured_threshold=threshold,
                    top_score=all_scored[0].score if all_scored else None,
                )

            ctx["after_llm_filter"] = len(final)
            ctx["threshold"]        = threshold

        return final
    