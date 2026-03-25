"""
app/server.py
-------------
FastAPI server that:
  1. Serves aura_rag_ui.html at /
  2. Exposes /api/health, /api/ingest, /api/query, /api/feedback
     — all delegating to api_client.py functions

Run:
    pip install fastapi uvicorn python-multipart
    uvicorn app.server:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import tempfile, shutil

from app.api_client import process_query, submit_feedback, run_ingestion, _get_pipeline

app = FastAPI(title="AURA-RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve UI ──────────────────────────────────────────────────────────────────

UI_FILE = os.path.join(os.path.dirname(__file__), "aura_rag_ui.html")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open(UI_FILE, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    try:
        pipeline = _get_pipeline()
        corpus = pipeline._faiss_store.get_all_texts()
        return {
            "status": "ok",
            "index_ready": len(corpus) > 0,
            "vectors": len(corpus),
        }
    except Exception as e:
        return {"status": "error", "index_ready": False, "vectors": 0, "detail": str(e)}


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/api/ingest")
async def ingest(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(80),
):
    tmp_dir = tempfile.mkdtemp(prefix="aura_upload_")
    paths = []
    try:
        for f in files:
            dest = os.path.join(tmp_dir, f.filename)
            with open(dest, "wb") as out:
                out.write(await f.read())
            paths.append(dest)

        result = run_ingestion(paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return JSONResponse(content=result)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

@app.post("/api/query")
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    result = process_query(req.query, session_id=req.session_id)
    return JSONResponse(content=result)


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    feedback_type: str           # "helpful" | "not_helpful" | "comment"
    feedback_text: Optional[str] = ""
    session_id: Optional[str]   = "default"

@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    result = submit_feedback(
        query=req.query,
        answer=req.answer,
        feedback_type=req.feedback_type,
        feedback_text=req.feedback_text or "",
        session_id=req.session_id,
    )
    return JSONResponse(content=result if isinstance(result, dict) else {"status": "ok"})
