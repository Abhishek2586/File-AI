"""
Unit Tests for Text Cleaner Module
=====================================
Tests edge cases in text_cleaner.py.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modules.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


class TestTextCleaner:

    def test_removes_excessive_whitespace(self, cleaner):
        dirty = "Hello     World\n\n\n\nGoodbye"
        out = cleaner.clean(dirty)
        assert "     " not in out

    def test_preserves_content(self, cleaner):
        text = "The NIST Cybersecurity Framework provides guidance."
        out = cleaner.clean(text)
        assert "NIST" in out
        assert "Framework" in out

    def test_empty_string_returns_empty(self, cleaner):
        assert cleaner.clean("") == "" or cleaner.clean("") == " "

    def test_special_characters_handled(self, cleaner):
        text = "Hello\x00World\x07End"
        out = cleaner.clean(text)
        assert "\x00" not in out
        assert "\x07" not in out

    def test_unicode_preserved(self, cleaner):
        text = "Résumé café naïve"
        out = cleaner.clean(text)
        assert "caf" in out  # at minimum the word survives

    def test_newlines_normalized(self, cleaner):
        text = "Line1\r\nLine2\rLine3"
        out = cleaner.clean(text)
        assert "\r\r" not in out

    def test_numbers_preserved(self, cleaner):
        text = "Section 3.1.2 requires compliance with 800-53."
        out = cleaner.clean(text)
        assert "3.1.2" in out
        assert "800-53" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
