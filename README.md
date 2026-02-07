# RAG-Lite

A lightweight, privacy-first Retrieval-Augmented Generation (RAG) system for knowledge-based question answering.

## Overview

RAG-Lite provides a modular RAG pipeline with hybrid search capabilities, running entirely on local infrastructure via Ollama. No data leaves your environment.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ Data Loader │ ──▶ │  Vector DB   │ ──▶ │  Retrieval  │ ──▶ │ Generation │
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘
                           │                    │
                    Embeddings (BGE)    Hybrid Search + Rerank
```

**Components:**

| Module | Description |
|--------|-------------|
| `data_loader` | Text file ingestion with UTF-8 encoding |
| `vector_db` | In-memory vector storage with embeddings |
| `retrieval` | Semantic search, BM25/TF-IDF, LLM reranking |
| `generation` | Context-aware response generation |
| `rag_pipeline` | Orchestration layer |
| `config` | Environment-based configuration |

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally
- 4GB+ RAM (model dependent)

## Installation

```bash
# Clone repository
git clone https://github.com/XingyuJinTI/rag-lite.git
cd rag-lite

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

**Pull models:**

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

## Configuration

All settings are configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` | Embedding model |
| `LANGUAGE_MODEL` | `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` | Generation model |
| `DATA_FILE` | `cat-facts.txt` | Input data file path |
| `OLLAMA_BASE_URL` | `None` | Custom Ollama endpoint |
| `USE_HYBRID_SEARCH` | `true` | Enable semantic + keyword search |
| `USE_RERANKING` | `false` | Enable LLM-based reranking |
| `USE_QUERY_EXPANSION` | `false` | Enable query expansion |
| `RETRIEVE_TOP_N` | `3` | Final results count |
| `RETRIEVE_K` | `20` | Candidates before reranking |
| `KEYWORD_METHOD` | `bm25` | Keyword method: `jaccard`, `tfidf`, `bm25` |
| `SEMANTIC_WEIGHT` | `0.7` | Semantic score weight |
| `KEYWORD_WEIGHT` | `0.3` | Keyword score weight |
| `BM25_K1` | `1.5` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 length normalization |

## Usage

**CLI:**

```bash
# Ensure Ollama is running
ollama serve

# Run application
python main.py
# or
rag-lite
```

**Programmatic:**

```python
from rag_lite.config import Config
from rag_lite.data_loader import load_text_file
from rag_lite.rag_pipeline import RAGPipeline

config = Config.from_env()
pipeline = RAGPipeline(config)

documents = load_text_file("your-data.txt")
pipeline.index_documents(documents)

results, response = pipeline.query("Your question here", stream=False)
print("".join(response))
```

## Security Considerations

### Data Privacy

- **Local Processing**: All inference runs locally via Ollama
- **No External APIs**: No data transmitted to third-party services
- **In-Memory Storage**: Data persists only during runtime
- **UTF-8 Encoding**: Consistent text handling

### Access Control

- Configure `OLLAMA_BASE_URL` for network-isolated deployments
- Environment variables for sensitive configuration
- No credential storage in codebase

### Logging

- Structured logging via Python `logging` module
- Configurable log levels
- No PII logged by default

### Recommendations for Production

1. Run Ollama behind a firewall or VPN
2. Use dedicated service accounts
3. Implement input validation for user queries
4. Monitor resource usage (memory, CPU)
5. Set appropriate file permissions on data files
6. Consider persistent vector storage for production workloads

## Retrieval Methods

**Hybrid Search** combines:
- **Semantic Search**: Cosine similarity on embeddings (default 70%)
- **Keyword Search**: BM25, TF-IDF, or Jaccard matching (default 30%)

**Reranking** (optional):
- LLM-based relevance scoring
- Improves precision at cost of latency

## Limitations

- In-memory storage (not suitable for large datasets)
- Single-threaded embedding generation
- No persistence across restarts

## License

MIT License
