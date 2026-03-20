"""
ChromaDB Handler Module
========================
Implements VectorDB interface using ChromaDB for persistent
local vector storage.

Features:
- Persistent storage to disk
- Collection management
- Similarity search with metadata filtering
- Duplicate detection
- Document-level operations
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.modules.vector_db_setup import VectorDB

logger = logging.getLogger(__name__)


class ChromaDBHandler(VectorDB):
    """
    ChromaDB implementation of the VectorDB interface.

    ChromaDB is the recommended choice for beginners:
    - No cloud account needed
    - Persistent storage on disk
    - Simple Python API
    - Built-in metadata filtering

    Attributes:
        persist_directory (str): Path to store ChromaDB data
        collection_name (str): Name of the collection
        client: ChromaDB client instance
        collection: Active ChromaDB collection
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "ai_file_assistant",
        dimension: int = 1536
    ):
        """
        Initialize ChromaDB handler.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name for the collection
            dimension: Embedding vector dimension (1536 for OpenAI)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.dimension = dimension
        self.client = None
        self.collection = None

        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # Create or get the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # cosine similarity
            )

            count = self.collection.count()
            logger.info(
                f"ChromaDB initialized: '{self.collection_name}' "
                f"({count} documents) at '{self.persist_directory}'"
            )

        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Run: pip install chromadb"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    def create_collection(self, name: str, dimension: int = 1536) -> bool:
        """Create a new collection (or switch to it if it exists)."""
        try:
            self.collection_name = name
            self.dimension = dimension
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{name}' ready")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            return False

    def _make_id(self, source_file: str, chunk_index: int) -> str:
        """Generate a deterministic, unique ID for a chunk."""
        raw = f"{source_file}::chunk_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        documents: List[str]
    ) -> bool:
        """
        Insert or update vectors with metadata.

        Args:
            ids: Unique IDs for each chunk
            vectors: Embedding vectors (length must equal 1536 for OpenAI)
            metadata: Metadata dicts per chunk
            documents: Original text for each chunk

        Returns:
            bool: Success status
        """
        if not ids:
            logger.warning("upsert called with empty ids list")
            return False

        try:
            # ChromaDB upsert adds new or replaces existing
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadata,
                documents=documents
            )
            logger.info(f"Upserted {len(ids)} vectors into ChromaDB")
            return True
        except Exception as e:
            logger.error(f"ChromaDB upsert failed: {e}")
            return False

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform similarity search.

        Args:
            vector: Query embedding
            top_k: Number of results to retrieve
            filter_metadata: Optional ChromaDB 'where' filter dict

        Returns:
            List of result dicts with id, document, metadata, score
        """
        try:
            count = self.collection.count()
            if count == 0:
                logger.warning("Query on empty collection")
                return []

            actual_k = min(top_k, count)

            query_params = {
                "query_embeddings": [vector],
                "n_results": actual_k,
                "include": ["documents", "metadatas", "distances"]
            }
            if filter_metadata:
                query_params["where"] = filter_metadata

            results = self.collection.query(**query_params)

            # Format results
            formatted = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                # Convert cosine distance → similarity score (0-1)
                score = 1.0 - distance

                formatted.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(score, 4)
                })

            logger.info(f"Query returned {len(formatted)} results")
            return formatted

        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by their IDs."""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def get_count(self) -> int:
        """Return total number of vectors stored."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"get_count failed: {e}")
            return 0

    def list_documents(self) -> List[str]:
        """Return sorted list of unique source document filenames."""
        try:
            count = self.collection.count()
            if count == 0:
                return []

            results = self.collection.get(include=["metadatas"])
            sources = set()
            for meta in results["metadatas"]:
                if meta and "source_file" in meta:
                    sources.add(meta["source_file"])
            return sorted(sources)
        except Exception as e:
            logger.error(f"list_documents failed: {e}")
            return []

    def delete_document(self, source_file: str) -> int:
        """
        Delete all chunks belonging to a source document.

        Args:
            source_file: Filename of the PDF (e.g., 'NIST_CSWP_29.pdf')

        Returns:
            int: Number of chunks deleted
        """
        try:
            results = self.collection.get(
                where={"source_file": source_file},
                include=["metadatas"]
            )
            ids_to_delete = results["ids"]
            if not ids_to_delete:
                logger.info(f"No chunks found for '{source_file}'")
                return 0

            self.collection.delete(ids=ids_to_delete)
            logger.info(
                f"Deleted {len(ids_to_delete)} chunks for '{source_file}'"
            )
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"delete_document failed: {e}")
            return 0

    def reset(self) -> bool:
        """Delete all vectors from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"reset failed: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Return collection metadata and statistics."""
        docs = self.list_documents()
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "total_chunks": self.get_count(),
            "total_documents": len(docs),
            "documents": docs,
            "dimension": self.dimension,
            "similarity_metric": "cosine"
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("ChromaDB Handler - Quick Test")
    print("=" * 60)

    db = ChromaDBHandler(persist_directory="./test_chroma_db")
    info = db.get_collection_info()
    print(f"\nCollection: {info['collection_name']}")
    print(f"Chunks stored: {info['total_chunks']}")
    print(f"Documents: {info['documents']}")

    # Test upsert with dummy embeddings
    dummy_vec = [0.1] * 1536
    ok = db.upsert(
        ids=["test_chunk_001"],
        vectors=[dummy_vec],
        metadata=[{"source_file": "test.pdf", "page_number": 1, "chunk_index": 0}],
        documents=["This is a test document for ChromaDB."]
    )
    print(f"\nUpsert success: {ok}")
    print(f"Total chunks now: {db.get_count()}")

    # Test query
    results = db.query(dummy_vec, top_k=1)
    if results:
        print(f"\nQuery result: {results[0]['document']}")
        print(f"Score: {results[0]['score']}")

    # Cleanup test data
    db.delete(["test_chunk_001"])
    print("\nTest chunk deleted. ChromaDB is working correctly!")
