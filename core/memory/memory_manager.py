"""
core/memory/memory_manager.py
------------------------------
Layer 5: Memory System

Wraps ChromaDB for episodic memory.

FIX APPLIED: Similarity threshold gate (reviewer fix #4).
Memory is ONLY injected if similarity >= 0.75.
This prevents bad past answers from polluting future generations.
"""

import uuid
from typing import List, Optional

from vectorstores.chroma_store import ChromaStore
from core.retriever.faiss_retriever import FaissRetriever
from app.config import CONFIG
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)


class MemoryManager:
    """
    Stores interactions after each pipeline run.
    Retrieves and injects relevant past context at query time.
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        retriever: FaissRetriever,
    ) -> None:
        self._store     = chroma_store
        self._retriever = retriever

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        query_id: str = "",
    ) -> str:
        """
        Returns a formatted memory context string to inject into the prompt,
        or an empty string if no relevant memory passes the threshold.
        """
        threshold = CONFIG.memory.similarity_threshold
        max_k     = CONFIG.memory.max_memory_results

        with log_stage(logger, "memory_recall", query_id=query_id) as ctx:
            emb = self._retriever.embed(query)
            memories = self._store.query_memory(
                query_embedding=emb,
                top_k=max_k,
                similarity_threshold=threshold,
            )
            ctx["memories_injected"] = len(memories)

        if not memories:
            return ""

        lines = []
        for m in memories:
            lines.append(
                f"[Past Q (sim={m['similarity']}): {m['query'][:80]}]\n"
                f"[Past A: {m['answer'][:200]}]"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        query: str,
        answer: str,
        chunks_used: List[str],
        feedback: str = "",
        feedback_category: str = "",
        confidence: float = 0.0,
        session_id: str = "",
    ) -> str:
        """Persist an interaction. Returns the generated interaction_id."""
        interaction_id = str(uuid.uuid4())
        emb = self._retriever.embed(query)
        self._store.store_interaction(
            interaction_id=interaction_id,
            query=query,
            query_embedding=emb,
            answer=answer,
            chunks_used=chunks_used,
            feedback=feedback,
            feedback_category=feedback_category,
            confidence=confidence,
            session_id=session_id,
        )
        return interaction_id
