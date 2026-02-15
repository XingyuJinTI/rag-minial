"""
Dataset loaders for RAG evaluation.

This module provides a flexible dataset abstraction that makes it easy to:
1. Swap between different datasets (cat-facts, RAGQArena)
2. Access ground truth for evaluation
3. Chunk documents appropriately per dataset type

Supported datasets:
- cat_facts: Simple text file with one fact per line (no ground truth Q&A)
- ragqa_arena: RAGQArena Tech dataset - tech Q&A with 28k+ documents
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document with optional metadata."""
    text: str
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = f"doc_{hash(self.text[:100]) % 10000:04d}"


@dataclass
class QAExample:
    """A question-answer example for evaluation."""
    question: str
    answer: str
    context: str  # The ground-truth passage that contains the answer
    example_id: str = ""
    gold_doc_ids: List[int] = field(default_factory=list)  # For RAGQArena
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.example_id:
            self.example_id = f"qa_{hash(self.question[:50]) % 10000:04d}"


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.
    
    Subclasses must implement:
    - load(): Load the dataset
    - get_documents(): Return documents for indexing
    - get_eval_examples(): Return QA examples for evaluation (if available)
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.documents: List[Document] = []
        self.eval_examples: List[QAExample] = []
        self._loaded = False
        self.config = kwargs
    
    @abstractmethod
    def load(self) -> "BaseDataset":
        """Load the dataset. Returns self for chaining."""
        pass
    
    def get_documents(self) -> List[Document]:
        """Get all documents for indexing."""
        if not self._loaded:
            self.load()
        return self.documents
    
    def get_chunks(self) -> List[str]:
        """Get document texts as a list of strings for indexing."""
        return [doc.text for doc in self.get_documents()]
    
    def get_eval_examples(self) -> List[QAExample]:
        """Get evaluation examples (question-answer pairs with ground truth)."""
        if not self._loaded:
            self.load()
        return self.eval_examples
    
    def has_ground_truth(self) -> bool:
        """Check if this dataset has ground truth for evaluation."""
        return len(self.get_eval_examples()) > 0
    
    def __len__(self) -> int:
        return len(self.get_documents())
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', documents={len(self.documents)}, eval_examples={len(self.eval_examples)})"


class CatFactsDataset(BaseDataset):
    """
    Simple cat facts dataset from a text file.
    One fact per line, no ground truth Q&A.
    
    Good for testing indexing and basic retrieval without evaluation metrics.
    """
    
    def __init__(self, file_path: str = "cat-facts.txt", **kwargs):
        super().__init__(name="cat_facts", **kwargs)
        self.file_path = file_path
    
    def load(self) -> "CatFactsDataset":
        """Load cat facts from text file."""
        path = Path(self.file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Cat facts file not found: {self.file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.documents = [
            Document(text=line, doc_id=f"cat_{i:04d}")
            for i, line in enumerate(lines)
        ]
        
        # No ground truth Q&A for this dataset
        self.eval_examples = []
        self._loaded = True
        
        logger.info(f"Loaded {len(self.documents)} cat facts")
        return self


class RAGQArenaDataset(BaseDataset):
    """
    RAGQArena Tech Dataset - Tech Q&A with corpus.
    
    A high-quality benchmark for RAG evaluation featuring:
    - 28,436 tech documents (corpus) - always fully indexed
    - 2,000+ Q&A pairs with ground truth responses and gold doc IDs
    - Natural language tech questions (StackOverflow-style)
    
    This dataset is well-suited for corpus-level retrieval evaluation
    because questions are specific and discriminating.
    
    Source: https://huggingface.co/dspy/cache
    
    Requires: pip install requests orjson
    
    Example:
        >>> # Full benchmark (all docs, all evals)
        >>> dataset = RAGQArenaDataset()
        >>> dataset.load()
        >>> chunks = dataset.get_chunks()  # 28k docs for indexing
        >>> examples = dataset.get_eval_examples()  # ~600 eval examples (dev split)
        
        >>> # Faster run with limited eval examples
        >>> dataset = RAGQArenaDataset(max_eval=100)
    """
    
    EXAMPLES_URL = "https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl"
    CORPUS_URL = "https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl"
    
    def __init__(
        self,
        max_eval: Optional[int] = None,
        max_doc_chars: int = 6000,
        cache_dir: Optional[str] = None,
        split_ratio: tuple = (0.2, 0.3, 0.5),  # train, dev, test
        use_split: str = "dev",  # Which split to use for evaluation
        **kwargs
    ):
        """
        Initialize RAGQArena dataset.
        
        Args:
            max_eval: Maximum evaluation examples (None = all ~600 in dev split)
            max_doc_chars: Truncate documents longer than this (default 6000)
            cache_dir: Directory to cache downloaded files (default: current dir)
            split_ratio: Tuple of (train, dev, test) ratios
            use_split: Which split to use for evaluation ("train", "dev", "test")
        """
        super().__init__(name="ragqa_arena", **kwargs)
        self.max_eval = max_eval
        self.max_doc_chars = max_doc_chars
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".")
        self.split_ratio = split_ratio
        self.use_split = use_split
    
    def _download_file(self, url: str, filename: str) -> Path:
        """Download file if not cached."""
        import requests
        
        filepath = self.cache_dir / filename
        
        if filepath.exists():
            logger.info(f"Using cached file: {filepath}")
            return filepath
        
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded to {filepath}")
        return filepath
    
    def load(self) -> "RAGQArenaDataset":
        """Load RAGQArena dataset from HuggingFace."""
        try:
            import orjson
        except ImportError:
            # Fallback to standard json
            import json as orjson
            orjson.loads = lambda x: __import__('json').loads(x)
        
        # Download files
        examples_path = self._download_file(self.EXAMPLES_URL, "ragqa_arena_tech_examples.jsonl")
        corpus_path = self._download_file(self.CORPUS_URL, "ragqa_arena_tech_corpus.jsonl")
        
        # Load corpus (documents for indexing) - always load ALL documents
        logger.info("Loading corpus (all documents)...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_lines = f.readlines()
        
        self.documents = []
        for i, line in enumerate(corpus_lines):
            if line.strip():
                item = orjson.loads(line)
                text = item.get('text', '')[:self.max_doc_chars]
                if text:
                    self.documents.append(Document(
                        text=text,
                        doc_id=f"ragqa_{i:05d}",
                        metadata={"source": "ragqa_arena_tech"}
                    ))
        
        logger.info(f"Loaded {len(self.documents)} corpus documents")
        
        # Load Q&A examples
        logger.info("Loading evaluation examples...")
        with open(examples_path, 'r', encoding='utf-8') as f:
            all_examples = [orjson.loads(line) for line in f if line.strip()]
        
        # Shuffle and split (reproducible)
        import random
        random.Random(42).shuffle(all_examples)
        
        n = len(all_examples)
        train_end = int(n * self.split_ratio[0])
        dev_end = train_end + int(n * self.split_ratio[1])
        
        splits = {
            "train": all_examples[:train_end],
            "dev": all_examples[train_end:dev_end],
            "test": all_examples[dev_end:],
        }
        
        examples = splits.get(self.use_split, splits["dev"])
        
        if self.max_eval and len(examples) > self.max_eval:
            examples = examples[:self.max_eval]
        
        # Convert to QAExample objects
        self.eval_examples = []
        for i, ex in enumerate(examples):
            # For RAGQArena, we use the response as context since
            # the ground truth is the response itself
            gold_ids = ex.get('gold_doc_ids', [])
            
            # Get actual ground truth context from corpus if available
            context = ""
            if gold_ids and self.documents:
                # Try to get text from first gold doc
                for gid in gold_ids:
                    if gid < len(self.documents):
                        context = self.documents[gid].text
                        break
            
            self.eval_examples.append(QAExample(
                question=ex['question'],
                answer=ex['response'],
                context=context if context else ex['response'],
                example_id=f"ragqa_qa_{i:05d}",
                gold_doc_ids=gold_ids,
                metadata={"source": "ragqa_arena_tech"}
            ))
        
        self._loaded = True
        
        logger.info(
            f"Loaded RAGQArena: {len(self.documents)} documents, "
            f"{len(self.eval_examples)} eval examples (split={self.use_split})"
        )
        return self
    
    def get_gold_documents(self, example: QAExample) -> List[Document]:
        """Get the gold (ground truth) documents for a Q&A example."""
        gold_docs = []
        for doc_id in example.gold_doc_ids:
            if doc_id < len(self.documents):
                gold_docs.append(self.documents[doc_id])
        return gold_docs


# Dataset registry for easy access
DATASET_REGISTRY = {
    "cat_facts": CatFactsDataset,
    "ragqa_arena": RAGQArenaDataset,
}


def load_dataset(
    name: str,
    **kwargs
) -> BaseDataset:
    """
    Load a dataset by name.
    
    Args:
        name: Dataset name (cat_facts or ragqa_arena)
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Loaded dataset instance
        
    Example:
        >>> # Load RAGQArena for evaluation
        >>> dataset = load_dataset("ragqa_arena", max_corpus=5000, max_eval=100)
        >>> chunks = dataset.get_chunks()
        >>> eval_examples = dataset.get_eval_examples()
        
        >>> # Load cat facts for simple testing
        >>> dataset = load_dataset("cat_facts", file_path="cat-facts.txt")
        >>> chunks = dataset.get_chunks()
    """
    if name in DATASET_REGISTRY:
        dataset_cls = DATASET_REGISTRY[name]
        if callable(dataset_cls) and not isinstance(dataset_cls, type):
            # It's a factory function
            dataset = dataset_cls(**kwargs)
        else:
            # It's a class
            dataset = dataset_cls(**kwargs)
        return dataset.load()
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )


def list_available_datasets() -> List[str]:
    """List all available dataset names."""
    return list(DATASET_REGISTRY.keys())
