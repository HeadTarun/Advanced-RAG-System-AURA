"""
embedder.py
-----------
Responsible for providing the embedding function used by FAISS and Chroma.
Wraps sentence-transformers/all-MiniLM-L6-v2 via LangChain's HuggingFace interface.
"""

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingGenerator:
    """
    Provides a reusable LangChain-compatible embedding model.

    Model: sentence-transformers/all-MiniLM-L6-v2
        - 384-dimensional dense vectors
        - Fast, lightweight, strong semantic quality
        - Runs locally (no API key required)

    The same EmbeddingGenerator instance should be shared across
    VectorIndexer to avoid loading the model multiple times.

    Does NOT store or retrieve. Single responsibility only.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        """
        Load the embedding model at construction time.
        Uses CPU by default; set model_kwargs={"device": "cuda"} for GPU.
        """
        print(f"[EmbeddingGenerator] Loading model: {self.MODEL_NAME}")

        self._model = HuggingFaceEmbeddings(
            model_name=self.MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # Cosine similarity ready
        )

        print("[EmbeddingGenerator] Model loaded successfully.")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the underlying LangChain embeddings object.

        This object can be passed directly to:
            - FAISS.from_texts(...)
            - Chroma.from_texts(...)

        Returns:
            HuggingFaceEmbeddings: LangChain-compatible embeddings instance.
        """
        return self._model
