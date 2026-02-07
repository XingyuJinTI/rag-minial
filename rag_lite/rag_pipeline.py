"""
Main RAG pipeline that orchestrates the entire retrieval and generation process.

This module provides a high-level interface for the complete RAG workflow.
"""

import logging
from typing import List, Tuple, Iterator, Optional

from .config import Config
from .vector_db import VectorDB
from .retrieval import retrieve, expand_query
from .generation import generate_response

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates retrieval and generation.
    
    This class provides a clean interface for:
    1. Loading and indexing documents
    2. Retrieving relevant context (semantic + BM25 with RRF fusion)
    3. Generating responses
    """

    def __init__(self, config: Config):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.vector_db = VectorDB(
            embedding_model=config.model.embedding_model,
            persist_directory=config.storage.persist_directory,
            collection_name=config.storage.collection_name,
            ollama_base_url=config.ollama_base_url
        )

    def index_documents(self, documents: List[str], show_progress: bool = True) -> None:
        """
        Index documents by creating embeddings and storing them.
        
        Args:
            documents: List of document chunks to index
            show_progress: Whether to log progress
        """
        logger.info(f"Indexing {len(documents)} documents...")
        self.vector_db.add_chunks(documents, show_progress=show_progress)
        logger.info(f"Indexed {self.vector_db.size()} documents")

    def retrieve(
        self, 
        query: str,
        top_n: Optional[int] = None,
        use_hybrid_search: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve relevant chunks for a query.
        
        When hybrid is enabled (default):
        1. Semantic search via ChromaDB HNSW
        2. BM25 keyword search
        3. RRF fusion
        4. Optional LLM reranking
        
        When hybrid is disabled:
        1. Semantic search only
        2. Optional LLM reranking
        
        Args:
            query: User query
            top_n: Number of results to return (overrides config)
            use_hybrid_search: Use hybrid search with RRF (overrides config)
            use_reranking: Whether to use reranking (overrides config)
            
        Returns:
            List of (chunk, score) tuples
        """
        top_n = top_n or self.config.retrieval.top_n
        hybrid = use_hybrid_search if use_hybrid_search is not None else self.config.retrieval.use_hybrid_search
        rerank = use_reranking if use_reranking is not None else self.config.retrieval.use_reranking
        
        return retrieve(
            query=query,
            vector_db=self.vector_db,
            language_model=self.config.model.language_model,
            top_n=top_n,
            retrieve_k=self.config.retrieval.retrieve_k,
            fusion_k=self.config.retrieval.fusion_k,
            use_hybrid_search=hybrid,
            use_reranking=rerank,
            bm25_k1=self.config.retrieval.bm25_k1,
            bm25_b=self.config.retrieval.bm25_b,
            rrf_k=self.config.retrieval.rrf_k,
            rerank_weight=self.config.retrieval.rerank_weight,
            original_score_weight=self.config.retrieval.original_score_weight,
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Tuple[str, float]],
        stream: bool = True
    ) -> Iterator[str]:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User query
            retrieved_chunks: List of (chunk, score) tuples from retrieval
            stream: Whether to stream the response
            
        Yields:
            Response text chunks (if streaming)
        """
        context_chunks = [chunk for chunk, _ in retrieved_chunks]
        return generate_response(
            query,
            context_chunks,
            self.config.model.language_model,
            stream=stream
        )

    def query(self, query: str, stream: bool = True) -> Tuple[List[Tuple[str, float]], Iterator[str]]:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            query: User query
            stream: Whether to stream the response
            
        Returns:
            Tuple of (retrieved_chunks, response_iterator)
        """
        retrieved = self.retrieve(query)
        response = self.generate(query, retrieved, stream=stream)
        return retrieved, response
