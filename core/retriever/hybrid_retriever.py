"""
core/retriever/hybrid_retriever.py
------------------------------------
Layer 2: Hybrid Retrieval

FIX APPLIED: async parallel execution of:
  - FAISS semantic search (3 query variants × 1 embedding call each)
  - BM25 keyword search (3 query variants)

All 6 searches run concurrently via asyncio.gather(), then merged
via Reciprocal Rank Fusion (RRF).

Why RRF?
  - Doesn't require score normalisation
  - Robust when score scales differ between FAISS and BM25
  - Consistently outperforms linear interpolation in benchmarks
"""

import asyncio
from typing import List, Tuple

from core.retriever.faiss_retriever import FaissRetriever
from core.retriever.keyword_retriever import KeywordRetriever
from core.query_processor import ProcessedQuery
from app.config import CONFIG
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists using RRF.
    Score for each doc = sum(1 / (k + rank)) across all lists it appears in.
    Higher is better.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc, _) in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged  # [(doc_text, rrf_score)]


class HybridRetriever:
    """
    Runs FAISS and BM25 in parallel across all query rewrites,
    then fuses with RRF.
    """

    def __init__(
        self,
        faiss_retriever: FaissRetriever,
        keyword_retriever: KeywordRetriever,
    ) -> None:
        self._faiss = faiss_retriever
        self._bm25  = keyword_retriever

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _async_faiss(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._faiss.retrieve, query, top_k
        )

    async def _async_bm25(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._bm25.retrieve, query, top_k
        )

    async def _retrieve_parallel(
        self, processed: ProcessedQuery, top_k: int
    ) -> List[List[Tuple[str, float]]]:
        """Fire all retrieval tasks concurrently."""
        queries = [
            processed.rewrites.specific,
            processed.rewrites.broad,
            processed.rewrites.keywords,
        ]
        tasks = []
        for q in queries:
            tasks.append(self._async_faiss(q, top_k))
            tasks.append(self._async_bm25(q, top_k))

        results = await asyncio.gather(*tasks)
        return list(results)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        processed: ProcessedQuery,
        query_id: str = "",
    ) -> List[Tuple[str, float]]:
        """
        Synchronous entry point — wraps the async implementation.
        Returns fused list of (chunk_text, rrf_score).
        """
        top_k_s = CONFIG.retrieval.top_k_semantic
        top_k_k = CONFIG.retrieval.top_k_keyword

        with log_stage(logger, "hybrid_retrieval", query_id=query_id) as ctx:
            # Run async gather in a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # In Jupyter / async context — use nest_asyncio or run_until_complete trick
                    import nest_asyncio
                    nest_asyncio.apply()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            all_results = asyncio.run(
                self._retrieve_parallel(processed, max(top_k_s, top_k_k))
            )

            fused = _reciprocal_rank_fusion(all_results, k=CONFIG.retrieval.rrf_k)
            ctx["total_candidates"] = len(fused)

        return fused
