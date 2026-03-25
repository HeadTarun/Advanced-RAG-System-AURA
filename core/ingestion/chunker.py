"""
chunker.py
----------
Responsible for splitting cleaned documents into semantically coherent chunks.
Attaches chunk_id and source metadata to every chunk produced.
"""

import uuid
from typing import List

from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter


class SemanticChunker:
    """
    Splits Document objects into overlapping chunks using
    RecursiveCharacterTextSplitter.

    Chunking strategy:
        - Split first at double newlines (paragraph boundaries).
        - Then at single newlines (line boundaries).
        - Then at sentence-ending punctuation.
        - Finally by character count as a fallback.

    Each chunk receives:
        - chunk_id:   unique identifier (UUID4)
        - source:     inherited from parent document metadata

    Does NOT embed or store. Single responsibility only.
    """

    # Separator priority: paragraph → line → sentence → word → character
    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        """
        Args:
            chunk_size:    Target character length per chunk.
            chunk_overlap: Character overlap between consecutive chunks
                           to preserve cross-boundary context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self._SEPARATORS,
            length_function=len,
            add_start_index=False,  # We use our own chunk_id scheme
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split all documents into chunks and attach metadata.

        Args:
            documents: Cleaned Document objects.

        Returns:
            List[Document]: Flat list of chunks with enriched metadata.
        """
        all_chunks: List[Document] = []

        for doc in documents:
            source = doc.metadata.get("source", "unknown")

            # Split this document into raw chunks
            raw_chunks = self._splitter.split_text(doc.page_content)

            for raw_text in raw_chunks:
                # Skip chunks that are only whitespace after splitting
                if not raw_text.strip():
                    continue

                chunk_metadata = {
                    **doc.metadata,           # Carry forward parent metadata
                    "source": source,         # Explicit for clarity
                    "chunk_id": str(uuid.uuid4()),  # Stable unique identifier
                }

                all_chunks.append(
                    Document(page_content=raw_text.strip(), metadata=chunk_metadata)
                )

        print(
            f"[SemanticChunker] Produced {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})."
        )
        return all_chunks
