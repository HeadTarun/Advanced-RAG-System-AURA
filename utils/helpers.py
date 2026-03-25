"""
utils/helpers.py
----------------
Shared utilities:
  - token counting
  - cosine similarity
  - LRU embedding cache
  - chunk deduplication
"""

import hashlib
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import tiktoken

# tiktoken encoder reused across calls
_ENCODER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Fast token count using tiktoken cl100k_base."""
    return len(_ENCODER.encode(text))


def truncate_to_budget(texts: List[str], budget: int) -> List[str]:
    """Return as many texts as fit within the token budget (greedy)."""
    kept, total = [], 0
    for t in texts:
        n = count_tokens(t)
        if total + n > budget:
            break
        kept.append(t)
        total += n
    return kept


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def deduplicate_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    threshold: float = 0.92,
) -> Tuple[List[str], List[List[float]]]:
    """
    Remove chunks whose embedding cosine similarity to any already-kept
    chunk exceeds `threshold`.  Returns (unique_chunks, unique_embeddings).
    """
    kept_chunks, kept_embs = [], []
    for chunk, emb in zip(chunks, embeddings):
        duplicate = any(
            cosine_similarity(emb, kept_e) >= threshold for kept_e in kept_embs
        )
        if not duplicate:
            kept_chunks.append(chunk)
            kept_embs.append(emb)
    return kept_chunks, kept_embs


# ---------------------------------------------------------------------------
# Embedding LRU cache (avoids re-embedding the same query string)
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbeddingCache:
    """Simple in-memory LRU cache for embeddings keyed by text hash."""

    def __init__(self, maxsize: int = 512):
        self._cache: dict = {}
        self._order: list = []
        self._maxsize = maxsize

    def get(self, text: str):
        key = _text_hash(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: List[float]) -> None:
        key = _text_hash(text)
        if key in self._cache:
            self._order.remove(key)
        elif len(self._order) >= self._maxsize:
            evict = self._order.pop(0)
            del self._cache[evict]
        self._cache[key] = embedding
        self._order.append(key)


# Singleton cache shared across retrievers
EMBEDDING_CACHE = EmbeddingCache(maxsize=1024)
