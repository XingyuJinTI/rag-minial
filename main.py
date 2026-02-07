#!/usr/bin/env python3
"""
Main entry point for RAG-Lite CLI application.

This script provides a command-line interface for the RAG system.
"""

import logging
import sys
from pathlib import Path

from rag_lite.config import Config
from rag_lite.data_loader import load_text_file
from rag_lite.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI."""
    # Load configuration
    try:
        config = Config.from_env()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        config = Config.default()
        logger.info("Using default configuration")
    
    # Load data
    data_file = Path(config.data_file)
    if not data_file.is_absolute():
        # Make relative to project root
        project_root = Path(__file__).parent
        data_file = project_root / data_file
    
    try:
        dataset = load_text_file(str(data_file))
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please ensure the data file exists or set DATA_FILE environment variable")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load data file: {e}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Index documents
    try:
        pipeline.index_documents(dataset, show_progress=True)
    except Exception as e:
        logger.error(f"Failed to index documents: {e}")
        sys.exit(1)
    
    # Interactive query loop
    print("\n" + "="*60)
    print("RAG-Lite: Knowledge-Based Question Answering")
    print("="*60)
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            query = input("Ask me a question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Retrieve and generate
            retrieved, response_stream = pipeline.query(query, stream=True)
            
            # Display retrieved knowledge
            print('\nRetrieved knowledge:')
            for chunk, score in retrieved:
                print(f' - (score: {score:.3f}) {chunk.strip()}')
            
            # Display response
            print('\nChatbot response:')
            for chunk in response_stream:
                print(chunk, end='', flush=True)
            print()  # New line at the end
            print()  # Extra spacing
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"An error occurred: {e}\n")


if __name__ == "__main__":
    main()
