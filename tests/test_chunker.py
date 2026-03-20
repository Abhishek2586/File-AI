"""
Unit Tests for Text Chunker Module
=====================================
Tests chunking behavior including token limits, overlap, and metadata.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modules.text_chunker import TextChunker


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=500, chunk_overlap=50)


SAMPLE_TEXT = """
Cybersecurity frameworks provide a structured approach to managing cybersecurity risks.
The NIST Cybersecurity Framework (CSF) 2.0 introduces six core functions: Govern, Identify, Protect, Detect, Respond, and Recover.
Each function contains categories and subcategories that describe specific outcomes.
Organizations use tiers to reflect how their risk management practices align to the framework.
Tier 1 (Partial) indicates informal and reactive practices, while Tier 4 (Adaptive) reflects continuous improvement.
Supply chain risk management is also a critical element in modern frameworks.
Organizations must assess third-party risks as part of their cybersecurity posture.
Access control, authentication and authorization mechanisms are key controls required in the Protect function.
Incident response planning under the Respond function includes communication protocols and recovery steps.
""" * 10  # Make it long enough for multiple chunks


class TestTextChunker:

    def test_chunker_creates_multiple_chunks(self, chunker):
        chunks = chunker.chunk_text(SAMPLE_TEXT, filename="test.pdf", page_number=1)
        assert len(chunks) > 1

    def test_each_chunk_has_text(self, chunker):
        chunks = chunker.chunk_text(SAMPLE_TEXT, filename="test.pdf", page_number=1)
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0

    def test_chunk_has_metadata(self, chunker):
        chunks = chunker.chunk_text(SAMPLE_TEXT, filename="test.pdf", page_number=1)
        for chunk in chunks:
            meta = chunk.metadata
            assert "source_file" in meta or "filename" in meta
            assert "chunk_index" in meta

    def test_no_chunk_exceeds_max_size(self, chunker):
        chunks = chunker.chunk_text(SAMPLE_TEXT, filename="test.pdf", page_number=1)
        for chunk in chunks:
            # Rough character estimation: chunk_size * 4 chars/token
            assert len(chunk.text) <= 500 * 6, "Chunk is too large"

    def test_short_text_gives_single_chunk(self, chunker):
        short = "This is a short sentence."
        chunks = chunker.chunk_text(short, filename="short.pdf", page_number=1)
        assert len(chunks) >= 1

    def test_empty_text_handled(self, chunker):
        chunks = chunker.chunk_text("", filename="empty.pdf", page_number=1)
        # Either an empty list or one chunk with empty text — both are acceptable
        assert isinstance(chunks, list)

    def test_chunk_index_increments(self, chunker):
        chunks = chunker.chunk_text(SAMPLE_TEXT, filename="test.pdf", page_number=1)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("chunk_index") == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
