"""
Vector database implementation for storing and managing embeddings.

This module provides an in-memory vector database for storing document chunks
and their corresponding embeddings.
"""

import logging
from typing import List, Tuple, Optional

import ollama

logger = logging.getLogger(__name__)


class VectorDB:
    """
    In-memory vector database for storing document chunks and embeddings.
    
    Each entry is stored as a tuple: (chunk: str, embedding: List[float])
    """

    def __init__(self, embedding_model: str, ollama_base_url: Optional[str] = None):
        """
        Initialize the vector database.
        
        Args:
            embedding_model: Name of the embedding model to use
            ollama_base_url: Optional base URL for Ollama API
        """
        self.embedding_model = embedding_model
        self._db: List[Tuple[str, List[float]]] = []
        
        # Configure Ollama client if base URL is provided
        if ollama_base_url:
            ollama.Client(host=ollama_base_url)

    def add_chunk(self, chunk: str) -> None:
        """
        Add a chunk to the database with its embedding.
        
        Args:
            chunk: Text chunk to add
        """
        try:
            embedding = ollama.embed(
                model=self.embedding_model, 
                input=chunk
            )['embeddings'][0]
            self._db.append((chunk, embedding))
        except Exception as e:
            logger.error(f"Failed to add chunk to database: {e}")
            raise

    def add_chunks(self, chunks: List[str], show_progress: bool = True) -> None:
        """
        Add multiple chunks to the database.
        
        Args:
            chunks: List of text chunks to add
            show_progress: Whether to log progress
        """
        total = len(chunks)
        for i, chunk in enumerate(chunks, 1):
            self.add_chunk(chunk)
            if show_progress:
                logger.info(f"Added chunk {i}/{total} to the database")

    def get_all(self) -> List[Tuple[str, List[float]]]:
        """
        Get all chunks and their embeddings.
        
        Returns:
            List of (chunk, embedding) tuples
        """
        return self._db.copy()

    def size(self) -> int:
        """Get the number of chunks in the database."""
        return len(self._db)

    def clear(self) -> None:
        """Clear all data from the database."""
        self._db.clear()
        logger.info("Vector database cleared")
