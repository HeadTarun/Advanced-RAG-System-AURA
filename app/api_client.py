"""
app/api_client.py
-----------------
Thin wrapper that connects the Streamlit UI to the AURA-RAG backend.
All backend logic lives in core/. This file only imports and delegates.

Exposes three functions:
    process_query()   → run a query through the full RAG pipeline
    submit_feedback() → send user feedback to the learning engine
    run_ingestion()   → ingest uploaded files into FAISS + Chroma
"""

import os
import sys

# Ensure project root is on the path so core.* imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import RAGPipeline, PipelineResult
from core.ingestion.pipeline import IngestionPipeline

import streamlit as st


# ── Singleton pipeline (cached across Streamlit reruns) ─────────────────────

@st.cache_resource(show_spinner=False)
def _get_pipeline() -> RAGPipeline:
    """Load RAGPipeline once; reuse across all sessions."""
    return RAGPipeline()


# ── Public API ───────────────────────────────────────────────────────────────

def process_query(query: str, session_id: str = "default") -> dict:
    """
    Run a user query through the full AURA-RAG pipeline.

    Returns:
        {
            "answer":     str,
            "confidence": float,      # 0.0 – 1.0
            "sources":    list[str],  # retrieved chunk texts
            "status":     "success" | "fallback",
        }
    """
    pipeline = _get_pipeline()
    result: PipelineResult = pipeline.query(query, session_id=session_id)

    return {
        "answer":     result.answer,
        "confidence": result.confidence,
        "sources":    result.chunks_used,
        "status":     "fallback" if not result.grounded else "success",
    }


def submit_feedback(
    query: str,
    answer: str,
    feedback_type: str,
    feedback_text: str = "",
    session_id: str = "default",
) -> dict:
    """
    Submit user feedback to the learning engine.

    Args:
        query:         The original user query.
        answer:        The answer that was rated.
        feedback_type: "helpful" | "not_helpful"
        feedback_text: Optional freeform comment.
        session_id:    Session identifier.

    Returns:
        { "category": str, "param_changes": dict }
    """
    pipeline = _get_pipeline()

    # Build a minimal PipelineResult so submit_feedback() has the right shape
    from core.pipeline import PipelineResult
    dummy_result = PipelineResult(
        query_id=session_id,
        query=query,
        answer=answer,
        confidence=0.0,
        grounded=(feedback_type == "helpful"),
        chunks_used=[],
    )

    combined_feedback = f"[{feedback_type}] {feedback_text}".strip()
    response = pipeline.submit_feedback(
        result=dummy_result,
        feedback_text=combined_feedback,
        session_id=session_id,
    )
    return response


def run_ingestion(
    file_paths: list,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> dict:
    """
    Ingest a list of file paths into FAISS + Chroma via the IngestionPipeline.

    Args:
        file_paths:    Absolute paths to PDF or TXT files on disk.
        chunk_size:    Target chunk character size.
        chunk_overlap: Overlap between chunks.

    Returns:
        { "status": "completed" | "error", "documents_processed": int, "message": str }
    """
    import tempfile, shutil

    # Write all files into a single temp directory so IngestionPipeline
    # can walk it with its directory-mode loader
    tmp_dir = tempfile.mkdtemp(prefix="aura_ingest_")
    try:
        for fp in file_paths:
            shutil.copy(fp, os.path.join(tmp_dir, os.path.basename(fp)))

        ingestion = IngestionPipeline(
            source_path=tmp_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            faiss_path="vectorstores/faiss_index",
            chroma_path="vectorstores/chroma_memory",
        )
        ingestion.run()

        return {
            "status": "completed",
            "documents_processed": len(file_paths),
            "message": f"Successfully ingested {len(file_paths)} file(s).",
        }

    except Exception as e:
        return {
            "status": "error",
            "documents_processed": 0,
            "message": str(e),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
