"""
core/retriever/faiss_retriever.py
---------------------------------
Semantic retrieval using FAISS + sentence-transformer embeddings.
Uses the EmbeddingCache to avoid re-embedding repeated queries.
"""

from typing import List, Tuple

from sentence_transformers import SentenceTransformer

from vectorstores.faiss_store import FaissStore
from utils.helpers import EMBEDDING_CACHE
from utils.logger import get_logger

logger = get_logger(__name__)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FaissRetriever:
    """
    Wraps FaissStore to provide embed-then-search in one call.
    Dependency-injected with a pre-built FaissStore instance.
    """

    def __init__(self, store: FaissStore, model_name: str = EMBED_MODEL_NAME) -> None:
        self._store = store
        self._model = SentenceTransformer(model_name)
        logger.info("faiss_retriever_ready", model=model_name)

    def embed(self, text: str) -> List[float]:
        """Embed a single string, using cache."""
        cached = EMBEDDING_CACHE.get(text)
        if cached is not None:
            return cached
        emb = self._model.encode(text, normalize_embeddings=True).tolist()
        EMBEDDING_CACHE.set(text, emb)
        return emb

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings, using cache for each."""
        return [self.embed(t) for t in texts]

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Returns (chunk, score) pairs sorted by descending similarity."""
        emb = self.embed(query)
        results = self._store.search(emb, top_k=top_k)
        logger.debug("faiss_retrieved", query=query[:60], results=len(results))
        return results
