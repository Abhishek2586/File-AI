"""
Unit Tests for PDF Processor Module
====================================
Tests for pdf_processor.py covering valid PDFs, corrupted inputs,
multi-page documents, and metadata extraction.
"""

import os
import sys
import pytest
import tempfile

# Adjust path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modules.pdf_processor import PDFProcessor


# --- Fixtures ---

@pytest.fixture
def processor():
    return PDFProcessor()

@pytest.fixture
def sample_pdf_path():
    """Returns path to a real NIST PDF if available."""
    candidates = [
        r"d:\Downloads\Infosys\ai-file-assistant\data\sample_pdfs\NIST_CSWP_29.pdf",
        r"d:\Downloads\Infosys\ai-file-assistant\data\sample_pdfs\NIST_IR_8596_iprd.pdf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    pytest.skip("No sample PDFs available for testing.")


# --- Tests ---

class TestPDFProcessor:

    def test_processor_initializes(self, processor):
        assert processor is not None

    def test_extract_text_from_valid_pdf(self, processor, sample_pdf_path):
        result = processor.extract_text_from_pdf(sample_pdf_path)
        assert result is not None
        assert "pages" in result
        assert result["total_pages"] > 0
        assert result["filename"] != ""

    def test_extracted_text_not_empty(self, processor, sample_pdf_path):
        result = processor.extract_text_from_pdf(sample_pdf_path)
        full_text = processor.get_full_text(result)
        assert len(full_text) > 100, "Extracted text should be non-trivial"

    def test_metadata_has_filename_and_pages(self, processor, sample_pdf_path):
        result = processor.extract_text_from_pdf(sample_pdf_path)
        assert "filename" in result
        assert "total_pages" in result
        assert isinstance(result["total_pages"], int)

    def test_nonexistent_file_raises(self, processor):
        with pytest.raises(Exception):
            processor.extract_text_from_pdf("/does/not/exist.pdf")

    def test_non_pdf_raises(self, processor):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is just a plain text file")
            tmp_path = f.name
        try:
            with pytest.raises(Exception):
                processor.extract_text_from_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_page_text_present_for_each_page(self, processor, sample_pdf_path):
        result = processor.extract_text_from_pdf(sample_pdf_path)
        for page in result["pages"]:
            assert "page_number" in page
            assert "text" in page


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
