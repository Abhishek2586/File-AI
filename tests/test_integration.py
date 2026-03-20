"""
Integration Tests - Complete End-to-End Flow
===============================================
Tests the complete pipeline: PDF Upload → Process → Store → Query → Answer.
As specified in Prompt 25.
"""

import os
import sys
import shutil
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modules.pdf_processor import PDFProcessor
from src.modules.text_cleaner import TextCleaner
from src.modules.text_chunker import TextChunker
from src.modules.embedding_pipeline import EmbeddingPipeline
from src.modules.openai_handler import OpenAIHandler
from src.modules.chromadb_handler import ChromaDBHandler
from src.modules.storage_pipeline import StoragePipeline
from src.modules.query_processor import QueryProcessor
from src.modules.context_builder import ContextBuilder
from src.modules.qa_system import QASystem

# --- Paths to real PDFs ---
SAMPLE_PDFS = {
    "nist_csf": r"d:\Downloads\Infosys\ai-file-assistant\data\sample_pdfs\NIST_CSWP_29.pdf",
    "nist_ir": r"d:\Downloads\Infosys\ai-file-assistant\data\sample_pdfs\NIST_IR_8596_iprd.pdf",
    "nist_sp": r"d:\Downloads\Infosys\ai-file-assistant\data\sample_pdfs\NIST_SP_500-291r2.pdf",
}

AVAILABLE_PDFS = [p for p in SAMPLE_PDFS.values() if os.path.exists(p)]


@pytest.fixture(scope="module")
def integration_db(tmp_path_factory):
    """A fully populated ChromaDB for integration tests."""
    tmpdir = tmp_path_factory.mktemp("int_db")
    db = ChromaDBHandler(persist_directory=str(tmpdir), collection_name="integration_test")
    handler = OpenAIHandler()
    pipeline = StoragePipeline(vector_db=db, openai_handler=handler)
    
    for pdf_path in AVAILABLE_PDFS:
        pipeline.process_pdf(pdf_path)
        
    return db


class TestEndToEndFlow:

    def test_pdfs_available(self):
        assert len(AVAILABLE_PDFS) > 0, "No sample PDFs available; cannot run integration tests."

    def test_database_is_populated(self, integration_db):
        count = integration_db.get_count()
        assert count > 0, "After ingestion, DB should have chunks"

    def test_list_documents_returns_all_ingested(self, integration_db):
        docs = integration_db.list_documents()
        for pdf_path in AVAILABLE_PDFS:
            filename = os.path.basename(pdf_path)
            assert filename in docs, f"Expected {filename} in indexed documents"

    def test_semantic_search_finds_correct_source(self, integration_db):
        handler = OpenAIHandler()
        qp = QueryProcessor(vector_db=integration_db, openai_handler=handler)
        results = qp.search("What are the five core functions of the NIST CSF?", top_k=5)
        assert len(results) > 0
        sources = [r["metadata"]["source_file"] for r in results]
        assert "NIST_CSWP_29.pdf" in sources

    def test_qa_answers_question_from_context(self, integration_db):
        handler = OpenAIHandler()
        qa = QASystem(vector_db=integration_db, openai_handler=handler, top_k=5)
        result = qa.answer_question("What is supply chain risk management?")
        assert result["answer"] != ""
        assert result["confidence"] > 0

    def test_qa_cites_sources(self, integration_db):
        handler = OpenAIHandler()
        qa = QASystem(vector_db=integration_db, openai_handler=handler, top_k=5)
        result = qa.answer_question("What is encryption?")
        assert len(result.get("sources", [])) > 0

    def test_qa_returns_no_results_gracefully(self, integration_db):
        """Query on a topic completely unrelated to the indexed PDFs."""
        handler = OpenAIHandler()
        qa = QASystem(vector_db=integration_db, openai_handler=handler, top_k=5, min_score=0.99)
        result = qa.answer_question("What is the recipe for chocolate chip cookies?")
        assert isinstance(result["answer"], str)

    def test_multi_turn_conversation(self, integration_db):
        handler = OpenAIHandler()
        qa = QASystem(vector_db=integration_db, openai_handler=handler, top_k=5)
        qa.answer_with_followup("What does the NIST CSF stand for?")
        result_2 = qa.answer_with_followup("What are its core functions?")
        assert result_2["answer"] != ""

    def test_context_builder_assembles_context(self, integration_db):
        handler = OpenAIHandler()
        qp = QueryProcessor(vector_db=integration_db, openai_handler=handler)
        results = qp.search("access control", top_k=5)
        cb = ContextBuilder(max_tokens=2000)
        ctx = cb.build_context(results)
        assert "Source:" in ctx or "NIST" in ctx or len(ctx) > 50

    def test_data_persistence_after_reload(self, tmp_path):
        """Simulates a restart by creating a new ChromaDB handler on the same path."""
        db1 = ChromaDBHandler(persist_directory=str(tmp_path), collection_name="persist_test")
        handler = OpenAIHandler()
        pipeline = StoragePipeline(vector_db=db1, openai_handler=handler)
        if AVAILABLE_PDFS:
            pipeline.process_pdf(AVAILABLE_PDFS[0])
        count_before = db1.get_count()
        
        # Reload from same path
        db2 = ChromaDBHandler(persist_directory=str(tmp_path), collection_name="persist_test")
        assert db2.get_count() == count_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
