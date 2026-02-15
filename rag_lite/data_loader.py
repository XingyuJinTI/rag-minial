"""Data loading utilities for RAG system."""

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# RAGQArena Tech corpus URL
RAGQA_CORPUS_URL = "https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl"


def load_text_file(file_path: str) -> List[str]:
    """Load text file, one document per line."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(lines)} documents from {file_path}")
    return lines


def load_jsonl_corpus(file_path: str, text_field: str = "text", max_chars: int = 6000) -> List[str]:
    """Load JSONL corpus file, extracting text field from each line."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")
    
    documents = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                text = item.get(text_field, "")[:max_chars]
                if text:
                    documents.append(text)
    
    logger.info(f"Loaded {len(documents)} documents from {file_path}")
    return documents


def load_ragqa_corpus(cache_dir: str = ".", max_chars: int = 6000) -> List[str]:
    """
    Load RAGQArena Tech corpus. Downloads if not cached.
    
    Args:
        cache_dir: Directory to cache the downloaded corpus
        max_chars: Truncate documents longer than this
    
    Returns:
        List of document texts
    """
    cache_path = Path(cache_dir) / "ragqa_arena_tech_corpus.jsonl"
    
    if not cache_path.exists():
        _download_file(RAGQA_CORPUS_URL, cache_path)
    
    return load_jsonl_corpus(str(cache_path), text_field="text", max_chars=max_chars)


def _download_file(url: str, dest: Path) -> None:
    """Download file from URL."""
    try:
        import requests
    except ImportError:
        raise ImportError("requests required for downloading. Install with: pip install requests")
    
    logger.info(f"Downloading {dest.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded to {dest}")
