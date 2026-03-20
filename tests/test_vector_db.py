"""
Unit Tests for Vector Database Modules
==========================================
Tests ChromaDB and FAISS handlers as per Prompt 24 requirements.
"""

import os
import sys
import shutil
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modules.chromadb_handler import ChromaDBHandler
from src.modules.faiss_handler import FAISSHandler
from src.modules.openai_handler import OpenAIHandler

# Shared embedding for speed
_handler = OpenAIHandler()
_EMBEDDING = _handler.get_embedding("cybersecurity framework test")
_DIM = len(_EMBEDDING)

# Dummy embeddings for tests that don't need a real one
DUMMY_VECTOR = [0.1] * _DIM


# ===== ChromaDB Tests =====

class TestChromaDBHandler:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.db = ChromaDBHandler(persist_directory=str(tmp_path), collection_name="unit_test")
        yield
        # No explicit teardown needed; tmp_path cleaned by pytest

    def test_initializes(self):
        assert self.db is not None

    def test_upsert_and_count(self):
        self.db.upsert(
            vectors=[DUMMY_VECTOR],
            metadata=[{"source_file": "a.pdf", "page_number": 1, "text": "hello"}],
            ids=["id_1"]
        )
        assert self.db.get_count() == 1

    def test_query_returns_results(self):
        self.db.upsert(
            vectors=[_EMBEDDING],
            metadata=[{"source_file": "a.pdf", "page_number": 1, "text": "cybersecurity framework test"}],
            ids=["id_2"]
        )
        results = self.db.query(_EMBEDDING, top_k=1)
        assert len(results) >= 1

    def test_delete_document(self):
        self.db.upsert(
            vectors=[DUMMY_VECTOR],
            metadata=[{"source_file": "delete_me.pdf", "page_number": 1, "text": "to be deleted"}],
            ids=["del_1"]
        )
        deleted = self.db.delete_document("delete_me.pdf")
        assert deleted >= 1
        assert self.db.get_count() == 0

    def test_list_documents(self):
        self.db.upsert(
            vectors=[DUMMY_VECTOR],
            metadata=[{"source_file": "doc1.pdf", "page_number": 1, "text": "test"}],
            ids=["doc_list_1"]
        )
        docs = self.db.list_documents()
        assert "doc1.pdf" in docs

    def test_duplicate_upsert_doesnt_increase_count(self):
        for _ in range(2):
            self.db.upsert(
                vectors=[DUMMY_VECTOR],
                metadata=[{"source_file": "dup.pdf", "page_number": 1, "text": "dup text"}],
                ids=["dup_id"]
            )
        assert self.db.get_count() == 1

    def test_empty_query_returns_empty(self):
        results = self.db.query(DUMMY_VECTOR, top_k=5)
        assert results == []


# ===== FAISS Tests =====

class TestFAISSHandler:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmpdir = str(tmp_path)
        self.db = FAISSHandler(
            index_path=os.path.join(self.tmpdir, "test.faiss"),
            metadata_path=os.path.join(self.tmpdir, "test_meta.json"),
            dimension=_DIM
        )
        yield

    def test_initializes(self):
        assert self.db is not None

    def test_upsert_and_count(self):
        self.db.upsert(
            vectors=[DUMMY_VECTOR],
            metadata=[{"source_file": "a.pdf", "page_number": 1, "text": "hello"}],
            ids=["id_faiss_1"]
        )
        assert self.db.get_count() == 1

    def test_query_returns_result(self):
        self.db.upsert(
            vectors=[_EMBEDDING],
            metadata=[{"source_file": "a.pdf", "page_number": 1, "text": "cybersecurity framework test"}],
            ids=["faiss_query_1"]
        )
        results = self.db.query(_EMBEDDING, top_k=1)
        assert len(results) >= 1

    def test_faiss_index_save_and_reload(self):
        self.db.upsert(
            vectors=[DUMMY_VECTOR],
            metadata=[{"source_file": "persist.pdf", "page_number": 1, "text": "saved data"}],
            ids=["persist_1"]
        )
        # Reload from same paths
        db2 = FAISSHandler(
            index_path=os.path.join(self.tmpdir, "test.faiss"),
            metadata_path=os.path.join(self.tmpdir, "test_meta.json"),
            dimension=_DIM
        )
        assert db2.get_count() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
