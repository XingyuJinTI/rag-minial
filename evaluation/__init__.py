"""
RAG-Lite Evaluation Package

This package provides evaluation tools for RAG systems, including:
- Dataset loaders (cat-facts, RAGQArena Tech)
- Retrieval metrics (Recall@K, MRR, NDCG, Hit Rate)
- Evaluation runner with comparison capabilities

Supported Datasets:
- cat_facts: Simple text file with one fact per line (no ground truth Q&A)
- ragqa_arena: RAGQArena Tech - 28k+ tech docs with Q&A pairs (recommended for evaluation)

Installation:
    pip install rag-lite[eval]
    # or
    pip install requests orjson

Quick Start:
    >>> from rag_lite import RAGPipeline, Config
    >>> from evaluation import load_dataset, RAGEvaluator
    >>>
    >>> # Load RAGQArena dataset (recommended for evaluation)
    >>> dataset = load_dataset("ragqa_arena", max_corpus=5000, max_eval=100)
    >>>
    >>> # Setup pipeline
    >>> config = Config.default()
    >>> pipeline = RAGPipeline(config)
    >>> pipeline.index_documents(dataset.get_chunks())
    >>>
    >>> # Evaluate
    >>> evaluator = RAGEvaluator(pipeline, dataset)
    >>> metrics = evaluator.evaluate()
    >>> evaluator.print_report()
"""

from .datasets import (
    load_dataset,
    list_available_datasets,
    BaseDataset,
    Document,
    QAExample,
    CatFactsDataset,
    RAGQArenaDataset,
)

from .metrics import (
    RetrievalMetrics,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    normalize_text,
    text_overlap_score,
)

from .evaluator import (
    RAGEvaluator,
    quick_evaluate,
)

__all__ = [
    # Dataset loading
    "load_dataset",
    "list_available_datasets",
    "BaseDataset",
    "Document",
    "QAExample",
    "CatFactsDataset",
    "RAGQArenaDataset",
    # Metrics
    "RetrievalMetrics",
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_ndcg_at_k",
    "normalize_text",
    "text_overlap_score",
    # Evaluator
    "RAGEvaluator",
    "quick_evaluate",
]
