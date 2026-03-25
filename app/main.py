"""
app/main.py
-----------
AURA-RAG: Adaptive Understanding & Retrieval Architecture

Entry point demonstrating the complete pipeline:
  1. Ingest sample documents via IngestionPipeline (Layer 0)
  2. Run queries through the full pipeline
  3. Submit user feedback → trigger learning
  4. Run RAGAS-style evaluation
  5. Print learning log to show self-improvement

Run:
    export GROQ_API_KEY=your_key_here
    python -m app.main
"""

import json
import os
import sys
import tempfile

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.pipeline import RAGPipeline
from core.ingestion.pipeline import IngestionPipeline
from utils.logger import setup_logging


# ---------------------------------------------------------------------------
# Sample corpus (replace with your actual documents in production)
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    # RAG
    "Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a language model. The retrieval step fetches relevant documents from an external knowledge base, which are then passed as context to the generator to produce grounded answers.",
    "RAG reduces hallucinations by grounding the language model's output in retrieved external knowledge rather than relying solely on parametric memory baked in during training.",
    "The key components of a RAG system are: a document store, an embedding model to encode queries and documents, a retrieval mechanism, and a language model for answer generation.",

    # FAISS
    "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It supports exact and approximate nearest neighbor search using methods like IVF (Inverted File Index) and HNSW.",
    "FAISS normalises vectors before indexing when using inner product (IP) indexes, which is equivalent to cosine similarity for normalised vectors. This makes it suitable for semantic search over sentence embeddings.",
    "FAISS stores vectors in memory and supports GPU acceleration. For production use, the IndexIVFFlat variant partitions the space into clusters, reducing search time by only scanning the nearest clusters.",

    # BM25 / Hybrid
    "BM25 (Best Match 25) is a bag-of-words retrieval function that scores documents based on the query terms appearing in each document, adjusted by term frequency and inverse document frequency (TF-IDF).",
    "Hybrid search combines dense retrieval (semantic, using embeddings) with sparse retrieval (keyword-based, like BM25). Dense retrieval handles synonyms and paraphrases; BM25 handles exact term matches and rare proper nouns.",
    "Reciprocal Rank Fusion (RRF) is a rank aggregation algorithm that merges multiple ranked lists without requiring score normalisation. Each document's final score is sum(1/(k+rank)) across all lists, where k is typically 60.",

    # Cross-encoders
    "Cross-encoders process the query and candidate document together in a single forward pass through the model, enabling full attention between them. This produces more accurate relevance scores than bi-encoders that encode query and document independently.",
    "In a two-stage retrieval pipeline, a fast bi-encoder (or BM25) first retrieves a large candidate set, then a cross-encoder reranks the top candidates. This balances speed and accuracy.",
    "The cross-encoder model cross-encoder/ms-marco-MiniLM-L-6-v2 is trained on the MS MARCO passage ranking dataset and provides strong reranking quality at small model size.",

    # Embeddings
    "Sentence-transformers/all-MiniLM-L6-v2 is a compact 384-dimensional sentence embedding model. It is fast, runs on CPU, and performs well on semantic similarity tasks, making it suitable for real-time RAG pipelines.",
    "Embedding models map variable-length text to fixed-size dense vectors in a semantic space. Texts with similar meanings are mapped to nearby vectors, enabling similarity search.",

    # LLM / Groq
    "Groq provides an LLM inference API using their Language Processing Unit (LPU) hardware, achieving very high token throughput compared to GPU-based inference. This makes it practical to use large models like LLaMA 3 70B in latency-sensitive pipelines.",
    "LLaMA 3 8B is a lightweight open-source language model suitable for classification, intent detection, query rewriting, and structured output generation. LLaMA 3 70B is a larger model with stronger reasoning capabilities for complex generation tasks.",

    # Confidence / Evaluation
    "Confidence evaluation in RAG involves checking whether every factual claim in the generated answer is directly supported by the retrieved context. This is different from general LLM confidence which is based on token probabilities.",
    "RAGAS is a framework for evaluating RAG pipelines using metrics like context precision, context recall, answer faithfulness, and answer relevance. These metrics can be computed without human annotation using LLM-based evaluation.",
    "Answer faithfulness measures the fraction of claims in the generated answer that are supported by the provided context. High faithfulness indicates low hallucination.",

    # Memory
    "ChromaDB is a vector database with built-in metadata filtering and collection management. In AURA-RAG it serves as the episodic memory store, persisting past interactions for similarity-based recall.",
    "Episodic memory in RAG systems stores past (query, answer, feedback) tuples. When a new query arrives, semantically similar past interactions are retrieved and injected into the context, enabling the system to learn from history.",
]

# Persistence paths — must match what RAGPipeline expects to load from
FAISS_PATH  = "vectorstores/faiss_index"
CHROMA_PATH = "vectorstores/chroma_memory"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_corpus_to_tempdir(documents: list[str]) -> str:
    """
    Write each document string to a numbered .txt file inside a temp directory.
    Returns the temp directory path for IngestionPipeline to consume.

    This is the bridge between the in-memory SAMPLE_DOCUMENTS list and the
    file-based IngestionPipeline, which expects PDF or TXT files on disk.
    In production, replace this with a real directory of source documents.
    """
    tmp_dir = tempfile.mkdtemp(prefix="aura_corpus_")
    for idx, text in enumerate(documents):
        file_path = os.path.join(tmp_dir, f"doc_{idx:04d}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    return tmp_dir


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  AURA-RAG — Self-Improving RAG System")
    print("=" * 60 + "\n")

    # --- Check API key ---
    if not os.getenv("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY environment variable not set.")
        print("  export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    # --- Initialize query pipeline ---
    print("[1/5] Initialising query pipeline...")
    pipeline = RAGPipeline()

    # ------------------------------------------------------------------
    # Layer 0: Ingestion Pipeline
    # Replaces the old pipeline.build_index(SAMPLE_DOCUMENTS) call.
    #
    # Flow:
    #   corpus .txt files
    #     → DocumentLoader → DocumentCleaner → SemanticChunker
    #     → EmbeddingGenerator
    #     → FAISS  (vectorstores/faiss_index/)   ← used by RAGPipeline retriever
    #     → Chroma (vectorstores/chroma_memory/) ← used by RAGPipeline memory
    # ------------------------------------------------------------------
    print("[2/5] Running ingestion pipeline (Layer 0)...")
    corpus_dir = _write_corpus_to_tempdir(SAMPLE_DOCUMENTS)

    ingestion = IngestionPipeline(
        source_path=corpus_dir,
        chunk_size=500,
        chunk_overlap=80,
        faiss_path=FAISS_PATH,
        chroma_path=CHROMA_PATH,
    )
    ingestion.run()

    print(f"      Ingested {len(SAMPLE_DOCUMENTS)} source documents.\n")

    # --- Run sample queries ---
    print("[3/5] Running sample queries...\n")
    sample_queries = [
        "What is retrieval-augmented generation and how does it reduce hallucinations?",
        "How does FAISS perform approximate nearest neighbor search?",
        "What is the difference between BM25 and dense retrieval, and when should I use hybrid search?",
        "Why use a cross-encoder for reranking?",
        "How does the AURA-RAG memory system work with ChromaDB?",
    ]

    results = []
    for i, q in enumerate(sample_queries, 1):
        print(f"  Query {i}: {q}")
        result = pipeline.query(q, session_id="demo_session")
        results.append(result)
        print(f"  Answer (confidence={result.confidence:.2f}, grounded={result.grounded}):")
        print(f"  {result.answer[:300]}{'...' if len(result.answer) > 300 else ''}")
        print()

    # --- Simulate user feedback on first result ---
    print("[4/5] Submitting feedback on query 1...")
    feedback_response = pipeline.submit_feedback(
     
    query_id=results[0].query_id,
    query=results[0].query,
    answer=results[0].answer,
    feedback="The answer was good but missed some detail about the retrieval step.",
    rating=4,  # optional (1–5 scale)
)
    
    print(f"  Feedback category: {feedback_response.category}")
    print(f"  Parameter changes: {feedback_response.param_changes}\n")

    # --- Run RAGAS evaluation ---
    print("[5/5] Running RAGAS-style evaluation on test suite...")
    with open("evaluation/test_queries.json") as f:
        test_cases = json.load(f)

    eval_results = pipeline.evaluate_batch(test_cases[:3])  

    print("\n  Evaluation Results:")
    print(f"  {'Query':<45} {'CP':>6} {'CR':>6} {'AF':>6} {'AR':>6} {'Comp':>6}")
    print("  " + "-" * 77)
    for r in eval_results:
        q = r["query"][:43] + ".." if len(r["query"]) > 43 else r["query"]
        print(
            f"  {q:<45} "
            f"{r['context_precision']:>6.2f} "
            f"{r['context_recall']:>6.2f} "
            f"{r['answer_faithfulness']:>6.2f} "
            f"{r['answer_relevance']:>6.2f} "
            f"{r['composite']:>6.2f}"
        )

    print("\n  CP=Context Precision, CR=Context Recall, AF=Answer Faithfulness, AR=Answer Relevance\n")

    print("=" * 60)
    print("  Pipeline run complete.")
    print("  Learning log: logs/learning_updates.jsonl")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()