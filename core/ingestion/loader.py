"""
loader.py
---------
Responsible for loading raw documents (PDF, TXT) from disk.
Outputs a flat list of LangChain Document objects for downstream processing.

PDF loading uses a fallback chain to handle the common LangChain/pypdf
version conflict where PyPDFLoader raises ImportError even when pypdf is
installed (caused by mismatched langchain-community vs pypdf versions):

    1. PyPDFLoader      (langchain-community, preferred)
    2. pypdf directly   (manual page extraction, no LangChain dependency)
    3. PyPDF2           (legacy fallback if pypdf API differs)
"""

import os
from typing import List

from langchain_classic.schema import Document


def _load_pdf_with_fallback(file_path: str) -> List[Document]:
    
    """
    Try three strategies in order to load a PDF.
    Returns a list of Document objects (one per page).
    Raises RuntimeError only if all three strategies fail.
    """

    # ── Strategy 1: LangChain PyPDFLoader (standard path) ───────────────
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_path
        return docs
    except Exception as e1:
        print(f"[DocumentLoader] PyPDFLoader failed ({e1}), trying pypdf directly…")

    # ── Strategy 2: pypdf directly (bypasses LangChain wrapper) ─────────
    try:
        import pypdf
        documents = []
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num},
                ))
        return documents
    except Exception as e2:
        print(f"[DocumentLoader] pypdf direct failed ({e2}), trying PyPDF2…")

    # ── Strategy 3: PyPDF2 (legacy, still common in older envs) ─────────
    try:
        import PyPDF2
        documents = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num},
                ))
        return documents
    except Exception as e3:
        raise RuntimeError(
            f"[DocumentLoader] All PDF strategies failed for {file_path}.\n"
            f"  PyPDFLoader : {e1}\n"
            f"  pypdf       : {e2}\n"
            f"  PyPDF2      : {e3}\n"
            f"Fix: pip install --upgrade pypdf langchain-community"
        )


class DocumentLoader:
    """
    Loads PDF and TXT files from a given directory or list of file paths.

    Responsibilities:
        - Detect file types and dispatch to the correct loader.
        - Return a unified list of Document objects regardless of source format.

    Does NOT clean, chunk, or embed. Single responsibility only.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, source_path: str):
        """
        Args:
            source_path: Path to a single file, or a directory of files to load.
        """
        self.source_path = source_path

    def _load_file(self, file_path: str) -> List[Document]:
        """Dispatch to the appropriate loader based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            # Use fallback chain — handles all pypdf/LangChain version combos
            return _load_pdf_with_fallback(file_path)

        elif ext == ".txt":
            try:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            except Exception:
                # Direct fallback: read file manually
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                docs = [Document(page_content=text, metadata={})]

            for doc in docs:
                doc.metadata["source"] = file_path
            return docs

        else:
            print(f"[DocumentLoader] Skipping unsupported file type: {file_path}")
            return []

    def load(self) -> List[Document]:
        """
        Load all supported documents from the configured source path.

        Returns:
            List[Document]: Flat list of loaded LangChain Document objects.
        """
        documents: List[Document] = []

        if os.path.isfile(self.source_path):
            documents = self._load_file(self.source_path)

        elif os.path.isdir(self.source_path):
            for root, _, files in os.walk(self.source_path):
                for fname in sorted(files):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in self.SUPPORTED_EXTENSIONS:
                        full_path = os.path.join(root, fname)
                        try:
                            docs = self._load_file(full_path)
                            documents.extend(docs)
                        except Exception as e:
                            # Log and continue — one bad file shouldn't stop the batch
                            print(f"[DocumentLoader] Skipping {fname}: {e}")
        else:
            raise FileNotFoundError(
                f"[DocumentLoader] Source path not found: {self.source_path}"
            )

        print(f"[DocumentLoader] Loaded {len(documents)} document pages/sections.")
        return documents