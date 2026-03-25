"""
core/retriever/keyword_retriever.py
------------------------------------
Sparse keyword retrieval using BM25.
Catches exact-match terms, product codes, proper nouns that semantic
search often misses.
"""

from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from utils.logger import get_logger

logger = get_logger(__name__)


class KeywordRetriever:
    """
    Build a BM25 index from a corpus, then score new queries against it.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus: List[str] = []

    def build(self, corpus: List[str]) -> None:
        """Tokenise and index the corpus."""
        self._corpus = corpus
        tokenised = [doc.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("bm25_built", docs=len(corpus))

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Returns (chunk_text, normalised_bm25_score) pairs.
        Scores are normalised to [0, 1] so they can be fused with FAISS scores.
        """
        if self._bm25 is None:
            logger.warning("bm25_not_initialized") 
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Normalise to [0, 1]
        max_score = float(np.max(scores)) if scores.max() > 0 else 1.0
        norm_scores = scores / max_score

        # Get top_k indices
        top_indices = np.argsort(norm_scores)[::-1][:top_k]
        results = [
            (self._corpus[int(i)], float(norm_scores[int(i)]))
            for i in top_indices
            if norm_scores[int(i)] > 0
        ]
        logger.debug("bm25_retrieved", query=query[:60], results=len(results))
        return results
