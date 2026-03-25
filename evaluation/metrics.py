"""
evaluation/metrics.py
----------------------
FIX APPLIED: Full RAGAS-style evaluation framework (reviewer fix #2).

Without this, the system cannot PROVE it is self-improving.

Metrics:
  1. Context Precision   — of retrieved chunks, what fraction are relevant?
  2. Context Recall      — of relevant chunks in ground truth, what fraction retrieved?
  3. Answer Faithfulness — is every claim in the answer supported by context?
  4. Answer Relevance    — does the answer actually address the query?

All metrics return float in [0, 1].  Higher = better.

Usage:
    evaluator = RAGEvaluator()
    scores = evaluator.evaluate(query, answer, contexts, ground_truth)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json

from sentence_transformers import SentenceTransformer
from utils.helpers import cosine_similarity
from llm.groq_client import groq_client
from utils.logger import get_logger

logger = get_logger(__name__)

_EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class EvalScores:
    context_precision:   float = 0.0
    context_recall:      float = 0.0
    answer_faithfulness: float = 0.0
    answer_relevance:    float = 0.0
    composite:           float = field(init=False)

    def __post_init__(self):
        self.composite = round(
            (self.context_precision
             + self.context_recall
             + self.answer_faithfulness
             + self.answer_relevance) / 4,
            4,
        )

    def to_dict(self) -> dict:
        return {
            "context_precision":   round(self.context_precision,   4),
            "context_recall":      round(self.context_recall,       4),
            "answer_faithfulness": round(self.answer_faithfulness,  4),
            "answer_relevance":    round(self.answer_relevance,     4),
            "composite":           self.composite,
        }


class RAGEvaluator:
    """
    Implements four RAGAS-inspired metrics using embedding similarity
    and LLM-based faithfulness checking.
    """

    # ------------------------------------------------------------------
    # 1. Context Precision
    # Measures: of retrieved chunks, how many are actually relevant?
    # Method: embed each chunk and the query; chunk is "relevant" if
    #         cosine similarity > threshold.
    # ------------------------------------------------------------------

    def context_precision(
        self,
        query: str,
        retrieved_chunks: List[str],
        threshold: float = 0.50,
    ) -> float:
        if not retrieved_chunks:
            return 0.0
        q_emb = _EMBED_MODEL.encode(query, normalize_embeddings=True).tolist()
        relevant = sum(
            1
            for c in retrieved_chunks
            if cosine_similarity(
                q_emb,
                _EMBED_MODEL.encode(c, normalize_embeddings=True).tolist()
            ) >= threshold
        )
        return relevant / len(retrieved_chunks)

    # ------------------------------------------------------------------
    # 2. Context Recall
    # Measures: of the ground-truth relevant passages, how many did we retrieve?
    # Method: embed each ground-truth passage; check if any retrieved chunk
    #         is "close enough" (cosine >= threshold).
    # ------------------------------------------------------------------

    def context_recall(
        self,
        retrieved_chunks: List[str],
        ground_truth_passages: List[str],
        threshold: float = 0.55,
    ) -> float:
        if not ground_truth_passages:
            return 1.0  # nothing to recall
        if not retrieved_chunks:
            return 0.0

        chunk_embs = [
            _EMBED_MODEL.encode(c, normalize_embeddings=True).tolist()
            for c in retrieved_chunks
        ]
        recalled = 0
        for gt in ground_truth_passages:
            gt_emb = _EMBED_MODEL.encode(gt, normalize_embeddings=True).tolist()
            if any(cosine_similarity(gt_emb, ce) >= threshold for ce in chunk_embs):
                recalled += 1
        return recalled / len(ground_truth_passages)

    # ------------------------------------------------------------------
    # 3. Answer Faithfulness
    # Measures: is every factual claim in the answer grounded in context?
    # Method: ask the 70B model (same as ConfidenceEvaluator) — reuse here
    #         for offline batch evaluation.
    # ------------------------------------------------------------------

    def answer_faithfulness(
        self,
        query: str,
        answer: str,
        context_chunks: List[str],
    ) -> float:
        context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a faithfulness evaluator. "
                    "Given CONTEXT and ANSWER, count: "
                    "(a) total factual claims in ANSWER, "
                    "(b) claims supported by CONTEXT. "
                    "Output ONLY valid JSON: "
                    '{"total_claims": <int>, "supported_claims": <int>}'
                ),
            },
            {
                "role": "user",
                "content": f"QUERY: {query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}",
            },
        ]
        try:
            result = groq_client.call_70b(prompt, json_mode=True, max_tokens=128)
            total     = int(result.get("total_claims",     1))
            supported = int(result.get("supported_claims", 0))
            return supported / max(total, 1)
        except Exception as e:
            logger.warning("faithfulness_eval_failed", error=str(e))
            return 0.0

    # ------------------------------------------------------------------
    # 4. Answer Relevance
    # Measures: does the answer actually address the original query?
    # Method: embed the answer and query; compute cosine similarity.
    #         A grounded but off-topic answer will score low here.
    # ------------------------------------------------------------------

    def answer_relevance(self, query: str, answer: str) -> float:
        q_emb = _EMBED_MODEL.encode(query,  normalize_embeddings=True).tolist()
        a_emb = _EMBED_MODEL.encode(answer, normalize_embeddings=True).tolist()
        return max(0.0, cosine_similarity(q_emb, a_emb))

    # ------------------------------------------------------------------
    # Combined evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query:                str,
        answer:               str,
        retrieved_chunks:     List[str],
        ground_truth_passages: Optional[List[str]] = None,
    ) -> EvalScores:
        """
        Run all four metrics and return an EvalScores dataclass.
        ground_truth_passages is optional — recall is skipped if absent.
        """
        cp  = self.context_precision(query, retrieved_chunks)
        cr  = self.context_recall(retrieved_chunks, ground_truth_passages or [])
        af  = self.answer_faithfulness(query, answer, retrieved_chunks)
        ar  = self.answer_relevance(query, answer)

        scores = EvalScores(
            context_precision=cp,
            context_recall=cr,
            answer_faithfulness=af,
            answer_relevance=ar,
        )
        logger.info("eval_scores", **scores.to_dict())
        return scores
