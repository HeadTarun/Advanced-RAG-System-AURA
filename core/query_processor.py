"""
core/query_processor.py
-----------------------
Layer 1: Query Processing

Uses LLaMA 3 8B to:
  1. Classify query intent
  2. Rewrite query in 3 forms (specific / broad / keywords)
  3. Return structured JSON (Pydantic-validated)

The three rewrites feed PARALLEL retrieval in the next layer,
which is the primary mechanism for improving recall.
"""

from pydantic import BaseModel, Field 
from typing import Optional
from llm.groq_client import groq_client
from llm.prompts import query_processor_prompt
from utils.logger import get_logger, log_stage

logger = get_logger(__name__)


class QueryRewrites(BaseModel):
    specific:  str
    broad:     str
    keywords:  str


class ProcessedQuery(BaseModel):
    original:   Optional[str] = ""
    intent:     str = Field(description="factual|analytical|conversational|comparative|procedural")
    rewrites:   QueryRewrites
    complexity: str = Field(description="simple|moderate|complex")


class QueryProcessor:
    """
    Single responsibility: transform a raw query into a structured
    ProcessedQuery with intent label and three retrieval-friendly rewrites.
    """

    def process(self, query: str, query_id: str = "") -> ProcessedQuery:
        with log_stage(logger, "query_processing", query_id=query_id) as ctx:
            messages = query_processor_prompt(query)
            result = groq_client.call_8b(
                messages,
                json_mode=True,
                schema=ProcessedQuery,
            )
            # inject original query (not in LLM output)
            result = ProcessedQuery(original=query, **result.model_dump(exclude={"original"}))
            ctx["intent"]     = result.intent
            ctx["complexity"] = result.complexity
        return result
