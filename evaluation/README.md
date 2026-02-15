# RAG-Lite Evaluation Suite

Comprehensive evaluation tools for measuring RAG retrieval performance.

## Installation

```bash
pip install -e .[eval]
# or
pip install requests orjson
```

## Datasets

| Dataset | Documents | Q&A Pairs | Use Case |
|---------|-----------|-----------|----------|
| `cat_facts` | ~100 | None | Quick testing, no metrics |
| `ragqa_arena` | 28,436 | ~2,000 | Full evaluation with metrics |

**RAGQArena Tech** (recommended) - Tech Q&A dataset with ground truth responses and gold document IDs. Ideal for corpus-level retrieval benchmarking.

## Quick Start

```bash
# Full benchmark (all 28k docs, all eval examples)
python -m evaluation.run_benchmark

# Faster run with limited eval examples
python -m evaluation.run_benchmark --max-eval 300

# Quick test with cat facts (no evaluation metrics)
python -m evaluation.run_benchmark --dataset cat_facts --file cat-facts.txt

# Skip interactive mode
python -m evaluation.run_benchmark --max-eval 100 --no-interactive
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of queries with relevant doc in top K |
| **MRR** | Mean Reciprocal Rank of first relevant result |
| **NDCG@10** | Normalized Discounted Cumulative Gain |
| **Hit Rate** | Fraction of queries with at least one hit |

## Comparing Configurations

Compare different retrieval strategies using config files:

```bash
# Compare RRF weight values
python -m evaluation.run_benchmark --config-file configs/rrf_weight_sweep.json --max-eval 100

# Compare retrieval strategies
python -m evaluation.run_benchmark --config-file configs/retrieval_strategies.json --max-eval 100

# Compare RRF k constant
python -m evaluation.run_benchmark --config-file configs/rrf_k_sweep.json --max-eval 100

# Use built-in comparison (semantic vs hybrid)
python -m evaluation.run_benchmark --compare --max-eval 100
```

## Config File Format

```json
{
  "configurations": [
    {"name": "semantic_only", "use_hybrid_search": false},
    {"name": "hybrid_default", "use_hybrid_search": true},
    {"name": "rrf_0.3", "rrf_weight": 0.3},
    {"name": "rrf_0.7", "rrf_weight": 0.7}
  ]
}
```

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_hybrid_search` | bool | true | Enable BM25 + semantic fusion |
| `use_reranking` | bool | false | Enable LLM reranking |
| `rrf_k` | int | 60 | RRF constant (higher = more uniform ranking) |
| `rrf_weight` | float | 0.7 | Semantic weight (0.0=all BM25, 1.0=all semantic) |
| `retrieve_k` | int | 50 | Candidates from each search method |
| `fusion_k` | int | 20 | Candidates after RRF fusion |
| `bm25_k1` | float | 1.5 | BM25 term frequency saturation |
| `bm25_b` | float | 0.75 | BM25 length normalization |
| `rerank_weight` | float | 0.6 | Weight for rerank score vs original |

## Pre-built Config Files

| File | Purpose |
|------|---------|
| `configs/rrf_weight_sweep.json` | Compare semantic vs BM25 balance |
| `configs/rrf_k_sweep.json` | Tune RRF ranking constant |
| `configs/retrieval_strategies.json` | Compare semantic/hybrid/rerank |
| `configs/bm25_tuning.json` | Tune BM25 parameters |
| `configs/candidate_pool.json` | Compare retrieval pool sizes |

## CLI Reference

```
python -m evaluation.run_benchmark --help

Options:
  --dataset, -d       Dataset: cat_facts, ragqa_arena (default: ragqa_arena)
  --file, -f          File path for cat_facts dataset
  --max-eval, -e      Maximum evaluation examples (default: all)
  --top-k, -k         Results to retrieve per query (default: 10)
  --config-file, -C   JSON config file for comparison
  --compare, -c       Use built-in comparison configs
  --no-hybrid         Disable hybrid search (semantic only)
  --rerank            Enable LLM reranking
  --no-interactive    Skip interactive query mode
```

## Programmatic Usage

```python
from rag_lite import RAGPipeline, Config
from evaluation import load_dataset, RAGEvaluator

# Full benchmark (all docs, all evals)
dataset = load_dataset("ragqa_arena")

# Or faster run with limited evals
dataset = load_dataset("ragqa_arena", max_eval=300)

# Setup pipeline
config = Config.default()
pipeline = RAGPipeline(config)
pipeline.index_documents(dataset.get_chunks())

# Evaluate
evaluator = RAGEvaluator(pipeline, dataset)
metrics = evaluator.evaluate(top_k=10)
evaluator.print_report()

# Compare configurations
configs = [
    {"name": "semantic_only", "use_hybrid_search": False},
    {"name": "hybrid", "use_hybrid_search": True},
]
results = evaluator.compare_configurations(configs)
```

## Output Example

```
======================================================================
RAG RETRIEVAL EVALUATION REPORT
======================================================================
Dataset: ragqa_arena
Total Queries: 300
Queries with Hits: 180
----------------------------------------------------------------------

RETRIEVAL METRICS:
  Recall@1:   0.3200
  Recall@3:   0.4600
  Recall@5:   0.5200
  Recall@10:  0.6000
  MRR:        0.4012
  Hit Rate:   0.6000
  NDCG@10:    0.4534
----------------------------------------------------------------------

PERFORMANCE:
  Avg Retrieval Time: 42.5ms
======================================================================
```
