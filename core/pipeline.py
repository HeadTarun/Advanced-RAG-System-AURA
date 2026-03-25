# core/pipeline.py
#
# CHANGES FROM PREVIOUS VERSION:
#
#   1. Added submit_feedback() public method.
#      main.py calls pipeline.submit_feedback(...) but the method was never
#      defined on RAGPipeline — only on the internal FeedbackClassifier /
#      LearningEngine components. This caused the AttributeError at [4/5].
#
#   2. Added FeedbackResult dataclass as the return type for submit_feedback().
#
#   3. All other logic unchanged.

import uuid
from dataclasses import dataclass
from typing import List, Optional

from core.query_processor import QueryProcessor
from core.retriever.faiss_retriever import FaissRetriever
from core.retriever.keyword_retriever import KeywordRetriever
from core.retriever.hybrid_retriever import HybridRetriever
from core.ranking.ranker import Ranker
from core.optimizer.context_optimizer import ContextOptimizer
from core.memory.memory_manager import MemoryManager
from core.generation.answer_generator import AnswerGenerator
from core.evaluation.confidence_evaluator import ConfidenceEvaluator
from core.failure.failure_handler import FailureHandler
from core.learning.feedback_classifier import FeedbackClassifier
from core.learning.learning_engine import LearningEngine

from vectorstores.faiss_store import FaissStore, FAISS_STORE_PATH
from vectorstores.chroma_store import ChromaStore

from app.config import CONFIG
from utils.logger import get_logger, log_stage, setup_logging

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    query_id: str
    query: str
    answer: str
    confidence: float
    grounded: bool
    chunks_used: List[str]
    attempt: int = 0


@dataclass
class FeedbackResult:
    query_id: str
    accepted: bool
    category: Optional[str] = None
    note: Optional[str] = None
    param_changes: Optional[dict] = None


class RAGPipeline:
    def __init__(self) -> None:
        setup_logging(
            CONFIG.observability.log_level,
            CONFIG.observability.log_format,
        )

        # FaissStore.load() is a classmethod: returns FaissStore | None
        loaded_store = FaissStore.load(FAISS_STORE_PATH)
        if loaded_store is not None:
            self._faiss_store = loaded_store
            logger.info("faiss_store_loaded_from_disk")
        else:
            self._faiss_store = FaissStore(dim=384)
            logger.warning("faiss_store_empty_no_index_found", path=FAISS_STORE_PATH)

        self._chroma_store = ChromaStore(CONFIG.memory.collection_name)

        # Retrievers
        self._faiss_retriever   = FaissRetriever(self._faiss_store)
        self._keyword_retriever = KeywordRetriever()

        # Build BM25 from restored FAISS text corpus
        corpus = self._faiss_store.get_all_texts()
        logger.info("bm25_corpus_size", size=len(corpus))
        if corpus:
            self._keyword_retriever.build(corpus)
        else:
            logger.warning(
                "bm25_not_built_empty_corpus",
                hint="Run ingestion to populate the FAISS index before querying.",
            )

        self._hybrid_retriever = HybridRetriever(
            self._faiss_retriever, self._keyword_retriever
        )

        # Other components
        self._query_processor      = QueryProcessor()
        self._ranker               = Ranker()
        self._optimizer            = ContextOptimizer(self._faiss_retriever)
        self._memory_manager       = MemoryManager(self._chroma_store, self._faiss_retriever)
        self._answer_generator     = AnswerGenerator()
        self._confidence_evaluator = ConfidenceEvaluator()
        self._failure_handler      = FailureHandler()
        self._feedback_classifier  = FeedbackClassifier()
        self._learning_engine      = LearningEngine()

        logger.info("pipeline_initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, user_query: str, session_id: str = "") -> PipelineResult:
        query_id   = str(uuid.uuid4())[:8]
        session_id = session_id or query_id

        with log_stage(logger, "full_pipeline", query_id=query_id):
            return self._run(user_query, query_id, session_id)

    def submit_feedback(
        self,
        query_id: str,
        query: str,
        answer: str,
        feedback: str,
        
        rating: Optional[int] = None,
    ) -> FeedbackResult:
        """
        Accept user feedback on a completed query and route it to the
        learning engine.

        Parameters
        ----------
        query_id : str   — the query_id from the PipelineResult being rated
        query    : str   — the original user query
        answer   : str   — the answer that was generated
        feedback : str   — free-text feedback ("too vague", "correct", …)
        rating   : int   — optional 1–5 star rating

        Returns
        -------
        FeedbackResult — whether accepted and what category it was classified into
        """
        try:
            category = self._feedback_classifier.classify(
                query=query,
                answer=answer,
                feedback=feedback,
            )
         

            changes = self._learning_engine.update(
                query_id=query_id,
                query=query,
                answer=answer,
                feedback=feedback,
                category=category,
                rating=rating,
            )

            logger.info(
                "feedback_accepted",
                query_id=query_id,
                category=category,
                rating=rating,
            )
            return FeedbackResult(
                query_id=query_id,
                accepted=True,
                category=category,
                param_changes=changes
            )

        except Exception as exc:
            # Feedback failures must never crash the caller
            logger.error("feedback_failed", query_id=query_id, error=str(exc))
            return FeedbackResult(
                query_id=query_id,
                accepted=False,
                note=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        user_query: str,
        query_id: str,
        session_id: str,
        attempt: int = 0,
    ) -> PipelineResult:
        processed = self._query_processor.process(user_query, query_id)

        candidates = self._hybrid_retriever.retrieve(processed, query_id)

        ranked_chunks = self._ranker.rank(user_query, candidates, query_id)

        if not ranked_chunks:
            return PipelineResult(
                query_id, user_query,
                "No relevant context found.",
                0.0, False, [], attempt,
            )

        chunk_texts = [c.text for c in ranked_chunks]

        context = self._optimizer.optimize(user_query, ranked_chunks, query_id)
        memory  = self._memory_manager.recall(user_query, query_id)

        answer = self._answer_generator.generate(
            user_query, context, memory, query_id
        )

        confidence = self._confidence_evaluator.evaluate(
            user_query, answer, context, query_id
        )

        return PipelineResult(
            query_id,
            user_query,
            answer,
            confidence.confidence,
            confidence.grounded,
            chunk_texts,
            attempt,
        )
        
    def evaluate_batch(self, test_cases: List[dict]) -> List[dict]:
        results = []

        for case in test_cases:
            query = case["query"]
            ground_truth = case.get("ground_truth", "")

            # Run pipeline
            result = self.query(query)

            retrieved_chunks = result.chunks_used
            answer = result.answer

            # --- Metrics ---
            context_precision = self._context_precision(retrieved_chunks, ground_truth)
            context_recall = self._context_recall(retrieved_chunks, ground_truth)
            answer_faithfulness = self._answer_faithfulness(answer, retrieved_chunks)
            answer_relevance = self._answer_relevance(answer, query)

            # Composite score
            composite = (
                context_precision
                + context_recall
                + answer_faithfulness
                + answer_relevance
            ) / 4

            results.append({
                "query": query,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_faithfulness": answer_faithfulness,
                "answer_relevance": answer_relevance,
                "composite": composite,
            })

        return results
    # ------------------------------------------------------------------
# Evaluation Metrics (RAGAS-style lite)
# ------------------------------------------------------------------

    def _context_precision(self, chunks, ground_truth):
        if not chunks:
            return 0.0

        relevant = sum(
            1 for c in chunks if ground_truth.lower() in c.lower()
        )
        return relevant / len(chunks)


    def _context_recall(self, chunks, ground_truth):
        if not ground_truth:
            return 0.0

        combined = " ".join(chunks).lower()
        return 1.0 if ground_truth.lower() in combined else 0.0


    def _answer_faithfulness(self, answer, chunks):
        result = self._confidence_evaluator.evaluate(
            query="evaluation",
            answer=answer,
            context_chunks=chunks,
        )
        return result.confidence


    def _answer_relevance(self, answer, query):
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        overlap = query_words.intersection(answer_words)
        return len(overlap) / max(len(query_words), 1)