"""
Utility functions for the backend
"""
import json
from typing import Any, Dict, List


def format_sources_json(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks as JSON string for database storage
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        JSON string
    """
    sources = []
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        sources.append({
            'text': chunk.get('text', '')[:200],  # Store first 200 chars
            'filename': metadata.get('filename', 'Unknown'),
            'chunk_index': metadata.get('chunk_index', 0),
            'pdf_id': metadata.get('pdf_id', None)
        })
    return json.dumps(sources)


def parse_sources_json(sources_json: str) -> List[Dict]:
    """
    Parse sources JSON from database
    
    Args:
        sources_json: JSON string from database
        
    Returns:
        List of source dictionaries
    """
    try:
        return json.loads(sources_json) if sources_json else []
    except:
        return []


def validate_role(role: str) -> bool:
    """
    Validate user role
    
    Args:
        role: Role string
        
    Returns:
        True if valid role
    """
    return role in ['student', 'professor']

