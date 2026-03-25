"""
cleaner.py
----------
Responsible for normalizing raw document text.
Removes noise without destroying semantic content.
"""

import re
from typing import List

from langchain_classic.schema import Document


class DocumentCleaner:
    """
    Cleans raw text in Document objects.

    Responsibilities:
        - Strip excessive whitespace (spaces, newlines, tabs).
        - Normalize unicode and punctuation artifacts.
        - Preserve all meaningful words, sentences, and structure.

    Does NOT chunk, embed, or modify metadata. Single responsibility only.
    """

    def _clean_text(self, text: str) -> str:
        """
        Apply cleaning rules to a single string.

        Rules (in order):
            1. Normalize unicode whitespace characters to standard space.
            2. Collapse runs of spaces/tabs into a single space.
            3. Collapse 3+ consecutive newlines into two (preserve paragraph breaks).
            4. Strip leading/trailing whitespace per line.
            5. Strip overall leading/trailing whitespace.
        """
        # 1. Normalize non-breaking spaces and other unicode whitespace to regular space
        text = text.replace("\xa0", " ").replace("\u200b", "")

        # 2. Collapse multiple spaces/tabs into one
        text = re.sub(r"[ \t]+", " ", text)

        # 3. Collapse 3+ blank lines into a clean paragraph break
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 4. Strip trailing space from each line
        text = "\n".join(line.strip() for line in text.split("\n"))

        # 5. Final strip
        text = text.strip()

        return text

    def clean(self, documents: List[Document]) -> List[Document]:
        """
        Clean the page_content of each Document in place (returns new list).

        Args:
            documents: Raw Document objects from the loader.

        Returns:
            List[Document]: Documents with cleaned text. Metadata is untouched.
        """
        cleaned: List[Document] = []

        for doc in documents:
            clean_text = self._clean_text(doc.page_content)

            # Skip documents that are empty after cleaning
            if not clean_text:
                print(
                    f"[DocumentCleaner] Skipping empty document from: "
                    f"{doc.metadata.get('source', 'unknown')}"
                )
                continue

            cleaned.append(Document(page_content=clean_text, metadata=doc.metadata))

        print(f"[DocumentCleaner] Cleaned {len(cleaned)} documents.")
        return cleaned
