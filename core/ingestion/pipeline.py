"""
pipeline.py
-----------
Orchestrates the full ingestion pipeline.

Flow:
    Raw Files
        → DocumentLoader     (load PDF / TXT)
        → DocumentCleaner    (normalize text)
        → SemanticChunker    (split + attach metadata)
        → EmbeddingGenerator (load model once)
        → VectorIndexer
            ├── FAISS  (text + embeddings → retrieval)
            └── Chroma (metadata + placeholders → memory)

The IngestionPipeline owns no logic of its own.
It wires the components together and runs them in strict order.
"""

from core.ingestion.loader import DocumentLoader
from core.ingestion.cleaner import DocumentCleaner
from core.ingestion.chunker import SemanticChunker
from core.ingestion.embedder import EmbeddingGenerator
from core.ingestion.indexer import VectorIndexer


class IngestionPipeline:
    """
    End-to-end ingestion pipeline for AURA-RAG.

    Wires DocumentLoader → DocumentCleaner → SemanticChunker
          → EmbeddingGenerator → VectorIndexer (FAISS + Chroma).

    Usage:
        pipeline = IngestionPipeline(source_path="data/docs/")
        pipeline.run()

    Configuration is passed at construction time.
    All heavy initialisation (model loading) happens in run() lazily.
    """

    def __init__(
        self,
        source_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        faiss_path: str = "vectorstores/faiss_index",
        chroma_path: str = "vectorstores/chroma_memory",
    ):
        """
        Args:
            source_path:   Path to a file or directory containing raw documents.
            chunk_size:    Target character size per chunk.
            chunk_overlap: Overlap between consecutive chunks (characters).
            faiss_path:    Persistence directory for the FAISS index.
            chroma_path:   Persistence directory for the Chroma store.
        """
        self.source_path = source_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_path = faiss_path
        self.chroma_path = chroma_path

    def run(self) -> None:
        """
        Execute the full ingestion pipeline in strict sequential order.

        Steps:
            1. Load raw documents from disk.
            2. Clean and normalize document text.
            3. Chunk documents into semantically coherent pieces.
            4. Load the embedding model (once).
            5. Index chunks into FAISS (retrieval store).
            6. Index chunks into Chroma (memory + metadata store).
        """
        print("\n" + "=" * 60)
        print("  AURA-RAG Ingestion Pipeline — Starting")
        print("=" * 60)

        # Step 1 — Load
        loader = DocumentLoader(source_path=self.source_path)
        documents = loader.load()

        if not documents:
            print("[IngestionPipeline] No documents loaded. Aborting.")
            return

        # Step 2 — Clean
        cleaner = DocumentCleaner()
        documents = cleaner.clean(documents)

        if not documents:
            print("[IngestionPipeline] All documents empty after cleaning. Aborting.")
            return

        # Step 3 — Chunk
        chunker = SemanticChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = chunker.chunk(documents)

        if not chunks:
            print("[IngestionPipeline] No chunks produced. Aborting.")
            return

        # Step 4 — Embeddings (model loaded once, shared across both stores)
        embedder = EmbeddingGenerator()
        embeddings = embedder.get_embeddings()

        # Step 5 — FAISS (retrieval)
        indexer = VectorIndexer(
            embeddings=embeddings,
            faiss_path=self.faiss_path,
            chroma_path=self.chroma_path,
        )
        indexer.index_faiss(chunks)

        # Step 6 — Chroma (memory + metadata)
        indexer.index_chroma(chunks)

        print("\n" + "=" * 60)
        print("  AURA-RAG Ingestion Pipeline — Complete")
        print(f"  Documents : {len(documents)}")
        print(f"  Chunks    : {len(chunks)}")
        print(f"  FAISS     : {self.faiss_path}")
        print(f"  Chroma    : {self.chroma_path}")
        print("=" * 60 + "\n")
