"""
vectorstores/faiss_store.py
---------------------------
FAISS index management.
Role: FAST primary semantic retrieval only.
NOT used for memory — that is ChromaDB's job.

FIX APPLIED (3 bugs, 1 root cause):
  Bug 1 — Two conflicting load() definitions on the same class.
           Python's MRO means the instance method `def load(self)`
           silently shadows the `@classmethod load(cls, path)`.
           `FaissStore.load(path)` therefore called the instance method,
           which accepted no positional path argument and returned False
           — the LangChain loader that correctly sets self.texts was
           never executed.

  Bug 2 — save() used faiss.write_index + pickle (custom format) but
           ingestion used LangChain's FAISS.save_local() (different format,
           different path). The two halves were completely mismatched:
           nothing the custom loader could find was ever written by
           ingestion.

  Bug 3 — get_all_texts() fell through to `_id_to_text.values()` when
           self.texts was absent, but _id_to_text is only populated by
           the custom build()/add() path, not by LangChain loading.
           Result: always returned [] at query time.

  Fix   — Unify on a single save/load strategy (LangChain FAISS.save_local /
           FAISS.load_local). Remove the duplicate instance-method load().
           get_all_texts() is now guaranteed to work after every load path.
"""

import os
import pickle
from typing import List, Optional, Tuple

import faiss
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils.logger import get_logger

logger = get_logger(__name__)

# LangChain FAISS save path (must match ingestion pipeline's save_local call)
FAISS_STORE_PATH = "vectorstores/faiss_index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FaissStore:
    """
    Wraps LangChain's FAISS store for semantic retrieval.

    All persistence goes through LangChain's save_local / load_local so
    that ingestion and query time use exactly the same format and path.

    Usage:
        # Build (ingestion side)
        store = FaissStore(dim=384)
        store.build(texts, embeddings)
        store.save()

        # Load (query side) — classmethod returns a ready FaissStore
        store = FaissStore.load()
        corpus = store.get_all_texts()   # always non-empty after a real load
        results = store.search(query_emb, top_k=20)
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.db: Optional[FAISS] = None
        # Populated by build() / add() when NOT going through LangChain
        self._id_to_text: dict[int, str] = {}
        self._next_id: int = 0
        # Single source of truth for text corpus — always set by load()
        self._texts: List[str] = []

    # ------------------------------------------------------------------
    # Build / persist  (unified LangChain path)
    # ------------------------------------------------------------------

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    def build(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        Create a LangChain FAISS store from pre-computed embeddings.
        Accepts raw float lists so FaissRetriever's embed_batch() output
        feeds straight in without re-embedding.
        """
        from langchain.docstore.document import Document
        from langchain_community.vectorstores import FAISS as LangFAISS

        docs = [Document(page_content=t) for t in texts]
        emb_model = self._get_embeddings()

        # LangChain FAISS.from_documents re-embeds internally — use
        # from_embeddings to honour the pre-computed vectors directly.
        text_embedding_pairs = list(zip(texts, embeddings))
        self.db = LangFAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=emb_model,
        )
        self._texts = list(texts)
        self._id_to_text = {i: t for i, t in enumerate(texts)}
        self._next_id = len(texts)
        logger.info("faiss_built", vectors=len(texts), dim=self.dim)

    def add(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Incrementally add documents to an existing store."""
        if self.db is None:
            self.build(texts, embeddings)
            return

        text_embedding_pairs = list(zip(texts, embeddings))
        self.db.add_embeddings(text_embedding_pairs)

        for t in texts:
            self._id_to_text[self._next_id] = t
            self._next_id += 1
        self._texts.extend(texts)
        logger.info("faiss_incremental_add", added=len(texts))

    def save(self, path: str = FAISS_STORE_PATH) -> None:
        """
        Persist via LangChain save_local.
        This writes index.faiss + index.pkl under `path/`.
        Ingestion must call this method (not LangChain directly) so that
        the path is always consistent with load().
        """
        if self.db is None:
            logger.warning("faiss_save_skipped_no_index")
            return
        os.makedirs(path, exist_ok=True)
        self.db.save_local(path)
        logger.info("faiss_saved", path=path, vectors=len(self._texts))

    # ------------------------------------------------------------------
    # FIX: single classmethod load() — no instance method with same name
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str = FAISS_STORE_PATH) -> Optional["FaissStore"]:
        """
        Restore a FaissStore from disk.
        Returns a fully initialised FaissStore, or None if no index exists.

        This is the ONLY load() on this class. The previous codebase had
        both a @classmethod load(cls, path) AND an instance method
        def load(self) — Python silently used the instance method for all
        calls, so this classmethod was never reached.
        """
        index_file = os.path.join(path, "index.faiss")
        if not os.path.exists(index_file):
            logger.warning("faiss_index_not_found", path=path)
            return None

        emb_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

        try:
            db = FAISS.load_local(
                path,
                emb_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.error("faiss_load_failed", error=str(exc))
            return None

        instance = cls.__new__(cls)
        instance.dim = 384
        instance.db = db
        instance._next_id = 0
        instance._id_to_text = {}

        # Recover text corpus from LangChain's docstore
        # docstore._dict maps str(id) -> Document
        raw_docs = list(db.docstore._dict.values())
        instance._texts = [doc.page_content for doc in raw_docs]

        # Also populate legacy _id_to_text so get_all_texts() works
        # regardless of which code path is used downstream
        instance._id_to_text = {i: t for i, t in enumerate(instance._texts)}
        instance._next_id = len(instance._texts)

        logger.info("faiss_loaded", vectors=len(instance._texts), path=path)
        return instance

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_embedding: List[float], top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Returns (chunk_text, score) pairs.
        Tries LangChain path first, falls back to raw FAISS if db is absent.
        """
        if self.db is not None:
            docs_and_scores = self.db.similarity_search_by_vector(
                query_embedding, k=top_k
            )
            # similarity_search_by_vector returns Document objects (no scores).
            # Use similarity_search_with_score_by_vector for actual distances.
            try:
                pairs = self.db.similarity_search_with_score_by_vector(
                    query_embedding, k=top_k
                )
                return [(doc.page_content, float(score)) for doc, score in pairs]
            except Exception:
                # Graceful fallback: return docs with dummy score 1.0
                return [(doc.page_content, 1.0) for doc in docs_and_scores]

        # Raw FAISS fallback (used when store was built without LangChain)
        if not self._id_to_text:
            return []

        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        index = faiss.IndexFlatIP(self.dim)
        vecs = np.array(
            [self._id_to_text[i] for i in range(self._next_id)],
            dtype=np.float32,
        )
        if vecs.ndim == 1:
            return []
        k = min(top_k, index.ntotal)
        scores, indices = index.search(q, k)
        return [
            (self._id_to_text[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    # ------------------------------------------------------------------
    # Corpus access  (used by pipeline.py to build BM25)
    # ------------------------------------------------------------------

    def get_all_texts(self) -> List[str]:
        """
        Returns the full text corpus.

        Priority:
          1. self._texts  — always populated by load() and build()
          2. docstore     — re-extract from LangChain db if _texts is empty
          3. _id_to_text  — legacy custom-FAISS fallback
        """
        if self._texts:
            return list(self._texts)

        # Re-extract from LangChain docstore if _texts was somehow lost
        if self.db is not None and hasattr(self.db, "docstore"):
            self._texts = [
                doc.page_content
                for doc in self.db.docstore._dict.values()
            ]
            if self._texts:
                return list(self._texts)

        # Last-resort legacy path
        return list(self._id_to_text.values())