"""
Query Processor Module
========================
Handles semantic search over the vector database.

Given a user question:
  1. Generates an embedding via OpenAI
  2. Performs cosine similarity search in the vector DB
  3. Ranks and returns the top-k most relevant document chunks

Designed to be database-agnostic (works with ChromaDB or FAISS).
"""

import logging
import time
from typing import List, Dict, Any, Optional

from src.modules.vector_db_setup import VectorDB
from src.modules.openai_handler import OpenAIHandler

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Semantic search processor for the AI File Assistant.

    Converts natural-language questions into vector queries,
    searches the vector database, and returns ranked results.

    Attributes:
        vector_db  : VectorDB instance (ChromaDB or FAISS)
        openai_handler : OpenAI handler for embedding generation
        default_top_k  : Default number of results to return
    """

    def __init__(
        self,
        vector_db: VectorDB,
        openai_handler: Optional[OpenAIHandler] = None,
        default_top_k: int = 5,
        min_score: float = 0.0
    ):
        """
        Initialize the query processor.

        Args:
            vector_db       : Configured VectorDB instance
            openai_handler  : Optional pre-built OpenAI handler
            default_top_k   : Default result count (5-10 recommended)
            min_score       : Minimum similarity score threshold (0-1)
        """
        self.vector_db = vector_db
        self.openai_handler = openai_handler or OpenAIHandler()
        self.default_top_k = default_top_k
        self.min_score = min_score

        # Session search stats
        self.search_stats = {
            "total_searches": 0,
            "total_results": 0,
            "total_search_time": 0.0,
            "cache_hits": 0
        }
        self._query_cache: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_documents(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find the most relevant document chunks for a question.

        Args:
            question      : Natural language question
            top_k         : Number of results (defaults to default_top_k)
            filter_source : Restrict results to a specific PDF filename
            use_cache     : Cache repeated identical queries

        Returns:
            List of result dicts, each containing:
              - id       : Chunk ID
              - document : Chunk text
              - metadata : Dict with source_file, page_number, chunk_index
              - score    : Cosine similarity score (0-1, higher = better)
              - rank     : 1-based rank position
        """
        if not question or not question.strip():
            logger.warning("Empty question received")
            return []

        question = question.strip()
        k = top_k or self.default_top_k

        # Cache lookup
        cache_key = f"{question}::{k}::{filter_source}"
        if use_cache and cache_key in self._query_cache:
            self.search_stats["cache_hits"] += 1
            logger.debug(f"Cache hit for: '{question[:50]}'")
            return self._query_cache[cache_key]

        start_time = time.time()

        try:
            # Step 1: Embed the question
            logger.info(f"Searching: '{question[:60]}'")
            query_vector = self.openai_handler.get_embedding(question)

            # Step 2: Vector similarity search
            filter_meta = {"source_file": filter_source} if filter_source else None
            raw_results = self.vector_db.query(
                vector=query_vector,
                top_k=k,
                filter_metadata=filter_meta
            )

            # Step 3: Apply minimum score filter and add rank
            results = []
            for rank, result in enumerate(raw_results, start=1):
                if result.get("score", 0) >= self.min_score:
                    result["rank"] = rank
                    results.append(result)

            elapsed = time.time() - start_time

            # Update stats
            self.search_stats["total_searches"] += 1
            self.search_stats["total_results"] += len(results)
            self.search_stats["total_search_time"] += elapsed

            logger.info(
                f"Search complete: {len(results)} results in {elapsed:.3f}s "
                f"(top score: {results[0]['score'] if results else 'N/A'})"
            )

            # Cache result
            if use_cache:
                self._query_cache[cache_key] = results

            return results

        except Exception as e:
            logger.error(f"Search failed for '{question[:50]}': {e}")
            return []

    def search_with_context(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search and return enriched result with query metadata.

        Returns:
            dict with:
              - question   : Original question
              - results    : List of ranked result dicts
              - total      : Number of results
              - search_time: Time taken in seconds
              - sources    : Unique source PDFs found
        """
        start = time.time()
        results = self.search_documents(question, top_k, filter_source)
        elapsed = time.time() - start

        sources = list({r["metadata"].get("source_file", "unknown")
                        for r in results if r.get("metadata")})

        return {
            "question": question,
            "results": results,
            "total": len(results),
            "search_time_seconds": round(elapsed, 3),
            "sources": sorted(sources)
        }

    def get_top_result(self, question: str) -> Optional[Dict[str, Any]]:
        """Return only the single best match, or None if DB is empty."""
        results = self.search_documents(question, top_k=1)
        return results[0] if results else None

    def get_stats(self) -> Dict[str, Any]:
        """Return search session statistics."""
        stats = self.search_stats.copy()
        n = stats["total_searches"]
        stats["avg_results_per_search"] = (
            round(stats["total_results"] / n, 1) if n else 0
        )
        stats["avg_search_time_ms"] = (
            round((stats["total_search_time"] / n) * 1000, 1) if n else 0
        )
        return stats

    def clear_cache(self):
        """Clear the query result cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")


# ------------------------------------------------------------------
# Module-level convenience function (matches PROMPT 12 spec)
# ------------------------------------------------------------------

def search_documents(
    question: str,
    vector_db: VectorDB,
    top_k: int = 5,
    openai_handler: Optional[OpenAIHandler] = None
) -> List[Dict[str, Any]]:
    """
    Search for relevant document chunks for a question.

    Args:
        question      : Natural language question
        vector_db     : Configured VectorDB instance
        top_k         : Results to return (default 5)
        openai_handler: Optional pre-built handler

    Returns:
        Ranked list of result dicts with id, document, metadata, score, rank
    """
    processor = QueryProcessor(
        vector_db=vector_db,
        openai_handler=openai_handler,
        default_top_k=top_k
    )
    return processor.search_documents(question)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from src.modules.chromadb_handler import ChromaDBHandler

    db = ChromaDBHandler()
    count = db.get_count()
    print(f"DB has {count} chunks in '{db.collection_name}' collection")

    if count == 0:
        print("No data — run storage_pipeline.py first to ingest PDFs.")
        sys.exit(0)

    processor = QueryProcessor(db)

    questions = [
        "What are the five functions of the NIST Cybersecurity Framework?",
        "How should organizations handle incident response?",
        "What is supply chain risk management?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        results = processor.search_documents(q, top_k=3)
        for r in results:
            src = r["metadata"].get("source_file", "?")
            pg  = r["metadata"].get("page_number", "?")
            print(f"  [{r['rank']}] score={r['score']:.3f} | {src} p.{pg}")
            print(f"       {r['document'][:120]}...")
