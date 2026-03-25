"""
llm/prompts.py
--------------
Single source of truth for every prompt in the system.
Prompts are plain functions that return a list[dict] (chat messages).
This makes them easy to test, version, and swap.
"""

from typing import List


# ===========================================================================
# 1. QUERY PROCESSING  (8B)
# ===========================================================================

def query_processor_prompt(query: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a query analysis assistant. "
                "Output ONLY valid JSON — no markdown, no explanation. "
                "Schema: "
                '{"intent": "factual|analytical|conversational|comparative|procedural", '
                '"rewrites": {"specific": "...", "broad": "...", "keywords": "..."}, '
                '"complexity": "simple|moderate|complex"}'
            ),
        },
        {
            "role": "user",
            "content": f"Analyse this query and return the JSON:\n\n{query}",
        },
    ]


# ===========================================================================
# 2. CHUNK RANKING  (70B)
# ===========================================================================

def ranker_prompt(query: str, chunk: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a relevance judge. "
                "Score how relevant the CHUNK is to the QUERY on a scale 1–10. "
                "Output ONLY valid JSON: "
                '{"score": <int 1-10>, "reason": "<one sentence>"}'
            ),
        },
        {
            "role": "user",
            "content": f"QUERY: {query}\n\nCHUNK:\n{chunk}",
        },
    ]


# ===========================================================================
# 3. CONTEXT SUMMARIZATION  (70B)
# ===========================================================================

def summarizer_prompt(query: str, chunk: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a context compressor. "
                "Summarize the CHUNK to only information relevant to the QUERY. "
                "Be concise. Return plain text — no JSON, no headers."
            ),
        },
        {
            "role": "user",
            "content": f"QUERY: {query}\n\nCHUNK:\n{chunk}",
        },
    ]


# ===========================================================================
# 4. ANSWER GENERATION  (70B)
# ===========================================================================

def answer_generator_prompt(
    query: str,
    context: str,
    memory_context: str = "",
) -> List[dict]:
    memory_block = (
        f"\n\nRELEVANT PAST INTERACTIONS:\n{memory_context}"
        if memory_context
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a precise, grounded assistant. "
                "Answer the QUERY using ONLY the provided CONTEXT. "
                "Do NOT introduce any information not in the context. "
                "For each factual claim, cite the source chunk index in [brackets]. "
                "If the context is insufficient, say: "
                "'I cannot answer this from the provided context.'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUERY: {query}\n\n"
                f"CONTEXT:\n{context}"
                f"{memory_block}"
            ),
        },
    ]


# ===========================================================================
# 5. CONFIDENCE EVALUATION  (70B)
# ===========================================================================

def confidence_evaluator_prompt(
    query: str, answer: str, context: str
) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a grounding auditor. "
                "Check whether every factual claim in ANSWER is directly supported "
                "by CONTEXT. "
                "Output ONLY valid JSON: "
                '{"confidence": <float 0.0-1.0>, '
                '"grounded": <bool>, '
                '"unsupported_claims": ["..."], '
                '"reasoning": "<brief>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUERY: {query}\n\n"
                f"ANSWER: {answer}\n\n"
                f"CONTEXT:\n{context}"
            ),
        },
    ]


# ===========================================================================
# 6. FEEDBACK CLASSIFICATION  (8B)
# ===========================================================================

def feedback_classifier_prompt(
    query: str, answer: str, feedback: str
) -> List[dict]:
    categories = (
        "correct | retrieval_error | ranking_error | reasoning_error | "
        "hallucination | vague_query | irrelevant_answer"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a failure analyst for a RAG system. "
                f"Classify the feedback into exactly one category: {categories}. "
                "Output ONLY valid JSON: "
                '{"category": "...", "confidence": <float 0-1>, "explanation": "..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUERY: {query}\n\n"
                f"ANSWER: {answer}\n\n"
                f"USER FEEDBACK: {feedback}"
            ),
        },
    ]
