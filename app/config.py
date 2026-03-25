"""
app/config.py
-------------
Central configuration for AURA-RAG.
All tunable parameters live here — the learning engine writes back to
LEARNING_STATE at runtime so parameter updates persist within a session.
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
MODEL_8B  = "llama-3.1-8b-instant"    # Fast: classify, rewrite, feedback
MODEL_70B = "llama-3.3-70b-versatile"   # Deep: rank, generate, evaluate

# ---------------------------------------------------------------------------
# Retrieval parameters  (learning engine may update these at runtime)
# ---------------------------------------------------------------------------
class RetrievalConfig(BaseModel):
    top_k_semantic:     int   = Field(default=20, description="FAISS top-k candidates")
    top_k_keyword:      int   = Field(default=20, description="BM25 top-k candidates")
    top_k_rerank:       int   = Field(default=8,  description="Cross-encoder top-k after merge")
    rrf_k:              int   = Field(default=60, description="RRF constant")

class RankingConfig(BaseModel):
    threshold:          float = Field(default=6.0,  description="Min score (1-10) to keep chunk")
    max_chunks_to_llm:  int   = Field(default=5,    description="Chunks passed to generator")

class ConfidenceConfig(BaseModel):
    min_confidence:     float = Field(default=0.70, description="Below this → retry/fallback")
    max_retries:        int   = Field(default=2)

class MemoryConfig(BaseModel):
    similarity_threshold: float = Field(default=0.75, description="Min sim to inject memory")
    max_memory_results:   int   = Field(default=3)
    collection_name:      str   = Field(default="aura_memory")

class TokenConfig(BaseModel):
    max_context_tokens: int   = Field(default=3000, description="Token budget for context")
    max_answer_tokens:  int   = Field(default=1024)

class ObservabilityConfig(BaseModel):
    log_level:      str  = Field(default="INFO")
    log_format:     str  = Field(default="json")   # "json" | "console"
    metrics_file:   str  = Field(default="logs/metrics.jsonl")


# ---------------------------------------------------------------------------
# Assembled global config — import this everywhere
# ---------------------------------------------------------------------------
class AuraConfig(BaseModel):
    retrieval:     RetrievalConfig     = RetrievalConfig()
    ranking:       RankingConfig       = RankingConfig()
    confidence:    ConfidenceConfig    = ConfidenceConfig()
    memory:        MemoryConfig        = MemoryConfig()
    tokens:        TokenConfig         = TokenConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

# Singleton — the learning engine mutates this object directly
CONFIG = AuraConfig()
