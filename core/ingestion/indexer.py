"""
indexer.py
----------
Responsible for persisting chunks to:
    - FAISS  → semantic retrieval (text + embeddings)
    - Chroma → memory and metadata (chunk_id, source, feedback placeholders)

FAISS and Chroma serve distinct purposes and must NOT be cross-used.
"""

import os
from typing import List

from langchain_classic.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorIndexer:
    """
    Writes chunked documents to FAISS and Chroma stores.

    FAISS store  → holds chunk text + embeddings; used by the retriever at query time.
    Chroma store → holds chunk_id, metadata, feedback_score, usage_count;
                   used by the memory/learning system at query time.

    Both stores are persisted to disk for reuse across sessions.

    Does NOT load files, clean, chunk, or embed. Single responsibility only.
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        faiss_path: str = "vectorstores/faiss_index",
        chroma_path: str = "vectorstores/chroma_memory",
    ):
        """
        Args:
            embeddings:  Shared LangChain embeddings instance.
            faiss_path:  Directory where the FAISS index will be saved.
            chroma_path: Directory where the Chroma collection will be persisted.
        """
        self.embeddings = embeddings
        self.faiss_path = faiss_path
        self.chroma_path = chroma_path

        os.makedirs(self.faiss_path, exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)

    # ------------------------------------------------------------------
    # FAISS — document retrieval store
    # ------------------------------------------------------------------

    def index_faiss(self, chunks: List[Document]) -> None:
        """
        Build a FAISS index from chunk texts and embeddings, then save to disk.

        Stores:  chunk text, chunk embedding vector
        Purpose: semantic similarity search at query time

        Args:
            chunks: Chunked Document objects (must have page_content).
        """
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        print(f"[VectorIndexer] Building FAISS index for {len(texts)} chunks...")

        faiss_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
        )

        faiss_store.save_local(self.faiss_path)
        print(f"[VectorIndexer] FAISS index saved → {self.faiss_path}")

    # ------------------------------------------------------------------
    # Chroma — memory and metadata store
    # ------------------------------------------------------------------

    def index_chroma(self, chunks: List[Document]) -> None:
        """
        Store chunk metadata in Chroma for memory and future learning.

        Stores:
            - chunk_id      → unique chunk identifier
            - source        → origin file path
            - feedback_score = 0  (placeholder; updated by feedback system)
            - usage_count    = 0  (placeholder; updated by query system)

        NOTE: Chroma embeddings are generated internally but only the metadata
        is what matters here. FAISS handles retrieval; Chroma handles memory.

        Args:
            chunks: Chunked Document objects (must have chunk_id in metadata).
        """
        texts = [chunk.page_content for chunk in chunks]

        # Build memory-ready metadata with learning placeholders
        metadatas = []
        for chunk in chunks:
            meta = {
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "source": chunk.metadata.get("source", "unknown"),
                "feedback_score": 0,   # Placeholder: updated by feedback classifier
                "usage_count": 0,      # Placeholder: incremented on each retrieval
            }
            metadatas.append(meta)

        print(f"[VectorIndexer] Persisting {len(texts)} chunks to Chroma...")

        Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.chroma_path,
            collection_name="aura_memory",
        )

        print(f"[VectorIndexer] Chroma store persisted → {self.chroma_path}")
