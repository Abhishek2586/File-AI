"""
Storage Pipeline Module
========================
Orchestrates the complete workflow: PDF → clean → chunk → embed → store.

This module ties together all Week 1-2 modules with the Week 3 vector
database to create a full document ingestion pipeline.

Features:
- End-to-end PDF processing and storage
- Progress tracking per document
- Duplicate detection and skip logic
- Statistics reporting
- Batch processing support
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from src.modules.pdf_processor import PDFProcessor
from src.modules.text_cleaner import TextCleaner
from src.modules.text_chunker import TextChunker
from src.modules.embedding_pipeline import EmbeddingPipeline
from src.modules.vector_db_setup import VectorDB

logger = logging.getLogger(__name__)


class StoragePipeline:
    """
    Full document ingestion pipeline: PDF → Vector DB.

    Stages:
        1. Extract text from PDF (PDFProcessor)
        2. Clean extracted text (TextCleaner)
        3. Split into chunks (TextChunker)
        4. Generate embeddings (EmbeddingPipeline)
        5. Store in vector DB with metadata (VectorDB)

    Attributes:
        vector_db: VectorDB instance (ChromaDB or FAISS)
        pdf_processor: PDF text extractor
        text_cleaner: Text preprocessor
        text_chunker: Text splitter
        embedding_pipeline: Embedding generator
        stats: Running statistics across all processed documents
    """

    def __init__(
        self,
        vector_db: VectorDB,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        openai_handler=None,
        skip_existing: bool = True
    ):
        """
        Initialize the storage pipeline.

        Args:
            vector_db: Configured VectorDB instance (ChromaDB or FAISS)
            chunk_size: Max characters per chunk
            chunk_overlap: Overlap between consecutive chunks
            openai_handler: Optional pre-configured OpenAI handler
            skip_existing: If True, skip PDFs already in the database
        """
        self.vector_db = vector_db
        self.skip_existing = skip_existing

        # Initialize sub-components
        self.pdf_processor = PDFProcessor()
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )

        # Embedding pipeline (use injected handler or create from .env)
        if openai_handler:
            self.embedding_pipeline = EmbeddingPipeline(
                openai_handler=openai_handler,
                use_cache=True,
                track_metrics=True
            )
        else:
            from src.modules.openai_handler import OpenAIHandler
            self.embedding_pipeline = EmbeddingPipeline(
                openai_handler=OpenAIHandler(),
                use_cache=True,
                track_metrics=True
            )

        # Session-level statistics
        self.stats = {
            "documents_processed": 0,
            "documents_skipped": 0,
            "documents_failed": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "total_time_seconds": 0.0,
            "details": []
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF and store its embeddings.

        Args:
            pdf_path: Absolute or relative path to the PDF file

        Returns:
            dict with keys: success, filename, chunks, embeddings,
                            time_seconds, error (if any)
        """
        filename = Path(pdf_path).name
        result = {
            "success": False,
            "filename": filename,
            "chunks": 0,
            "embeddings": 0,
            "time_seconds": 0.0,
            "error": None
        }

        start = time.time()

        # 1. Duplicate check
        if self.skip_existing:
            existing = self.vector_db.list_documents()
            if filename in existing:
                logger.info(f"Skipping '{filename}' (already in DB)")
                result["error"] = "already_in_db"
                self.stats["documents_skipped"] += 1
                return result

        try:
            # 2. Extract text
            logger.info(f"[1/4] Extracting text from '{filename}'")
            extracted = self.pdf_processor.extract_text_from_pdf(pdf_path)

            if not extracted["success"]:
                raise RuntimeError(
                    extracted.get("error", "PDF extraction failed")
                )

            # If PDF has no pages, work with mock text (for testing)
            pages = extracted.get("pages", [])
            if not pages:
                raise RuntimeError("No pages extracted from PDF")

            # 3. Clean + chunk per page
            logger.info(f"[2/4] Cleaning and chunking text")
            all_chunks = []
            for page in pages:
                page_text = self.text_cleaner.clean(page["text"])
                if not page_text.strip():
                    continue
                chunks = self.text_chunker.chunk_text(
                    page_text,
                    source_file=filename,
                    page_number=page["page_number"]
                )
                all_chunks.extend(chunks)

            if not all_chunks:
                raise RuntimeError("No usable chunks after cleaning")

            result["chunks"] = len(all_chunks)
            logger.info(f"    Created {len(all_chunks)} chunks")

            # 4. Generate embeddings
            logger.info(f"[3/4] Generating embeddings")
            embedding_results = \
                self.embedding_pipeline.generate_embeddings_for_chunks(
                    all_chunks,
                    show_progress=True
                )
            result["embeddings"] = len(embedding_results)
            logger.info(f"    Generated {len(embedding_results)} embeddings")

            # 5. Store in vector DB
            logger.info(f"[4/4] Storing in vector database")
            ids, vectors, metadatas, documents = [], [], [], []

            for idx, (chunk, embedding) in enumerate(embedding_results):
                chunk_id = self._make_chunk_id(filename, idx)
                ids.append(chunk_id)
                vectors.append(embedding)

                meta = {
                    "source_file": filename,
                    "page_number": chunk.metadata.get("page_number", 0),
                    "chunk_index": idx,
                    "chunk_size": len(chunk.text),
                    "total_chunks": len(embedding_results),
                    "ingested_at": datetime.now().isoformat()
                }
                metadatas.append(meta)
                documents.append(chunk.text)

            ok = self.vector_db.upsert(ids, vectors, metadatas, documents)
            if not ok:
                raise RuntimeError("Vector DB upsert failed")

            result["success"] = True
            result["time_seconds"] = round(time.time() - start, 2)

            self.stats["documents_processed"] += 1
            self.stats["total_chunks"] += result["chunks"]
            self.stats["total_embeddings"] += result["embeddings"]
            self.stats["total_time_seconds"] += result["time_seconds"]

            logger.info(
                f"✓ '{filename}' stored: {result['chunks']} chunks "
                f"in {result['time_seconds']}s"
            )

        except Exception as e:
            result["error"] = str(e)
            result["time_seconds"] = round(time.time() - start, 2)
            self.stats["documents_failed"] += 1
            logger.error(f"✗ Failed to process '{filename}': {e}")

        self.stats["details"].append(result)
        return result

    def process_pdfs(self, pdf_files: List[str]) -> Dict[str, Any]:
        """
        Process a list of PDF files and store all embeddings.

        Args:
            pdf_files: List of PDF file paths

        Returns:
            dict with overall statistics and per-document details
        """
        logger.info(f"Starting batch: {len(pdf_files)} PDF(s)")

        for pdf_path in pdf_files:
            if not Path(pdf_path).exists():
                logger.warning(f"File not found: '{pdf_path}' — skipping")
                self.stats["documents_failed"] += 1
                self.stats["details"].append({
                    "success": False,
                    "filename": Path(pdf_path).name,
                    "error": "file_not_found"
                })
                continue

            self.process_pdf(pdf_path)

        self._print_summary()
        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Return current pipeline statistics."""
        stats = self.stats.copy()
        stats["database_info"] = self.vector_db.get_collection_info() \
            if hasattr(self.vector_db, "get_collection_info") else {}
        stats["embedding_metrics"] = \
            self.embedding_pipeline.get_performance_metrics()
        return stats

    def get_database_info(self) -> Dict[str, Any]:
        """Return vector DB collection info."""
        if hasattr(self.vector_db, "get_collection_info"):
            return self.vector_db.get_collection_info()
        return {
            "total_chunks": self.vector_db.get_count(),
            "documents": self.vector_db.list_documents()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk_id(filename: str, chunk_index: int) -> str:
        """Deterministic ID so re-ingesting replaces, not duplicates."""
        import hashlib
        raw = f"{filename}::chunk_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _print_summary(self):
        """Print a summary table to stdout."""
        print("\n" + "=" * 65)
        print("STORAGE PIPELINE SUMMARY")
        print("=" * 65)
        print(f"  Documents processed : {self.stats['documents_processed']}")
        print(f"  Documents skipped   : {self.stats['documents_skipped']}")
        print(f"  Documents failed    : {self.stats['documents_failed']}")
        print(f"  Total chunks stored : {self.stats['total_chunks']}")
        print(f"  Total embeddings    : {self.stats['total_embeddings']}")
        print(
            f"  Total time          : "
            f"{self.stats['total_time_seconds']:.1f}s"
        )
        print("-" * 65)
        for d in self.stats["details"]:
            status = "✓" if d.get("success") else "✗"
            extra = (
                f"{d.get('chunks', 0)} chunks, "
                f"{d.get('embeddings', 0)} embeddings"
                if d.get("success")
                else d.get("error", "unknown error")
            )
            print(f"  {status} {d.get('filename', 'unknown')}: {extra}")
        print("=" * 65)


# ------------------------------------------------------------------
# Standalone convenience function (used by tests / CLI)
# ------------------------------------------------------------------

def process_and_store_pdfs(
    pdf_files: List[str],
    vector_db: VectorDB,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> Dict[str, Any]:
    """
    Convenience function: process PDFs and store embeddings.

    Args:
        pdf_files: List of PDF file paths
        vector_db: Configured VectorDB instance
        chunk_size: Characters per chunk
        chunk_overlap: Overlap characters

    Returns:
        Statistics dict from StoragePipeline.get_stats()
    """
    pipeline = StoragePipeline(
        vector_db=vector_db,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return pipeline.process_pdfs(pdf_files)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print("=" * 65)
    print("Storage Pipeline Module - Demo")
    print("=" * 65)

    from src.modules.chromadb_handler import ChromaDBHandler

    db = ChromaDBHandler(persist_directory="./demo_chroma_db")
    pipeline = StoragePipeline(vector_db=db)

    print(f"\nDatabase currently holds: {db.get_count()} chunks")
    print(f"Known documents: {db.list_documents()}")

    # Run with actual PDFs if provided via CLI
    if len(sys.argv) > 1:
        pdfs = sys.argv[1:]
        results = pipeline.process_pdfs(pdfs)
    else:
        print("\nUsage: python modules/storage_pipeline.py path/to/doc.pdf ...")
        print("Or use process_and_store_pdfs() in your code.")
