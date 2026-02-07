"""
Data loading utilities for reading datasets.

This module handles loading text data from files for use in the RAG system.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_text_file(file_path: str) -> List[str]:
    """
    Load text file and return lines as a list.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of lines from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
        logger.info(f'Loaded {len(lines)} entries from {file_path}')
        return lines
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise
