"""
Unit Tests for QA System Module
====================================
Tests for qa_system.py covering answer generation, source citation,
confidence scores, and multi-turn conversation.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modules.chromadb_handler import ChromaDBHandler
from src.modules.openai_handler import OpenAIHandler
from src.modules.qa_system import QASystem

# Load one PDF for embedding into a shared DB fixture
SAMPLE_PDF = r"d:\Downloads\Infosys\ai-file-assistant\data\sample_pdfs\NIST_CSWP_29.pdf"


@pytest.fixture(scope="module")
def qa_db(tmp_path_factory):
    """A ChromaDB populated with one real PDF."""
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("NIST_CSWP_29.pdf not available for QA unit tests.")
    from src.modules.storage_pipeline import StoragePipeline
    tmpdir = tmp_path_factory.mktemp("qa_unit_db")
    db = ChromaDBHandler(persist_directory=str(tmpdir), collection_name="qa_test")
    handler = OpenAIHandler()
    pipeline = StoragePipeline(vector_db=db, openai_handler=handler)
    pipeline.process_pdf(SAMPLE_PDF)
    return db


@pytest.fixture
def qa(qa_db):
    handler = OpenAIHandler()
    return QASystem(vector_db=qa_db, openai_handler=handler, top_k=5)


class TestQASystem:

    def test_qa_initializes(self, qa):
        assert qa is not None

    def test_answer_question_returns_dict(self, qa):
        result = qa.answer_question("What is the NIST CSF?")
        assert isinstance(result, dict)
        assert "answer" in result

    def test_answer_is_non_empty(self, qa):
        result = qa.answer_question("What are the core functions of NIST CSF?")
        assert len(result["answer"].strip()) > 10

    def test_sources_are_returned(self, qa):
        result = qa.answer_question("What does the Protect function involve?")
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_confidence_score_in_range(self, qa):
        result = qa.answer_question("What is the Identify function?")
        conf = result.get("confidence", 0)
        assert 0.0 <= conf <= 1.0

    def test_empty_db_handled_gracefully(self, tmp_path):
        handler = OpenAIHandler()
        empty_db = ChromaDBHandler(persist_directory=str(tmp_path), collection_name="empty_qa_test")
        qa_empty = QASystem(vector_db=empty_db, openai_handler=handler)
        result = qa_empty.answer_question("What is the NIST CSF?")
        assert isinstance(result["answer"], str)  # Should not throw

    def test_multi_turn_conversation(self, qa):
        qa.reset_conversation()
        r1 = qa.answer_with_followup("What is the NIST CSF?")
        r2 = qa.answer_with_followup("What is the Detect function?")
        assert r2["answer"] != ""

    def test_session_stats_track_questions(self, qa):
        qa.reset_conversation()
        qa.answer_question("What are cybersecurity tiers?")
        qa.answer_question("What is the Govern function?")
        stats = qa.get_session_stats()
        assert stats.get("questions_answered", 0) >= 2

    def test_answer_with_followup_returns_same_shape(self, qa):
        result = qa.answer_with_followup("What is supply chain risk management?")
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
