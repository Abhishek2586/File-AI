"""
Vector Database Setup Module
==============================
Defines abstract base class and factory for vector databases.

Supports:
- ChromaDB (recommended, local)
- FAISS (alternative, fast)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VectorDB(ABC):
    """
    Abstract base class for vector database implementations.
    
    All vector databases must implement these core methods to
    provide a unified interface for the AI File Assistant.
    """

    @abstractmethod
    def create_collection(self, name: str, dimension: int = 1536) -> bool:
        """
        Create a new collection/index in the vector database.

        Args:
            name (str): Collection name
            dimension (int): Embedding dimension (1536 for OpenAI ada-002)

        Returns:
            bool: True if created successfully
        """
        pass

    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[List[float]],
               metadata: List[Dict[str, Any]], documents: List[str]) -> bool:
        """
        Insert or update vectors with metadata.

        Args:
            ids: Unique identifiers for each vector
            vectors: Embedding vectors
            metadata: Metadata dicts (source, page, chunk_index, etc.)
            documents: Original text for each vector

        Returns:
            bool: True if upserted successfully
        """
        pass

    @abstractmethod
    def query(self, vector: List[float], top_k: int = 5,
              filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for nearest vectors.

        Args:
            vector: Query embedding
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of dicts with keys: id, document, metadata, score
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by ID."""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Return total number of vectors stored."""
        pass

    @abstractmethod
    def list_documents(self) -> List[str]:
        """Return list of unique source document filenames."""
        pass

    @abstractmethod
    def delete_document(self, source_file: str) -> int:
        """Delete all chunks from a given source PDF."""
        pass

    @abstractmethod
    def reset(self) -> bool:
        """Delete all vectors from the collection."""
        pass


def get_vector_db(db_type: str = "chromadb", **kwargs) -> VectorDB:
    """
    Factory method to create a vector database instance.

    Args:
        db_type: "chromadb" or "faiss"
        **kwargs: Configuration passed to the chosen database

    Returns:
        VectorDB: Configured database instance

    Examples:
        >>> db = get_vector_db("chromadb", persist_directory="./chroma_db")
        >>> db = get_vector_db("faiss", index_path="./faiss_index")
    """
    db_type = db_type.lower()

    if db_type == "chromadb":
        from src.modules.chromadb_handler import ChromaDBHandler
        return ChromaDBHandler(**kwargs)
    elif db_type == "faiss":
        from src.modules.faiss_handler import FAISSHandler
        return FAISSHandler(**kwargs)
    else:
        raise ValueError(f"Unsupported vector database: '{db_type}'. "
                         f"Choose from: 'chromadb', 'faiss'")
