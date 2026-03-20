"""
Unit Tests for OpenAI Handler Module
========================================
Tests embedding generation and Q&A via the FastRouter API proxy.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modules.openai_handler import OpenAIHandler


@pytest.fixture
def handler():
    return OpenAIHandler()


class TestOpenAIHandler:

    def test_handler_initializes(self, handler):
        assert handler is not None
        assert handler.client is not None

    def test_get_embedding_returns_list(self, handler):
        embedding = handler.get_embedding("NIST cybersecurity framework")
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_embedding_has_correct_dimension(self, handler):
        embedding = handler.get_embedding("test text")
        # text-embedding-ada-002 / 3-small both output vectors > 256 dims
        assert len(embedding) >= 256

    def test_embedding_values_are_floats(self, handler):
        embedding = handler.get_embedding("cybersecurity")
        assert all(isinstance(v, float) for v in embedding)

    def test_two_different_texts_give_different_embeddings(self, handler):
        emb1 = handler.get_embedding("cybersecurity framework")
        emb2 = handler.get_embedding("banana smoothie recipe")
        assert emb1 != emb2

    def test_generate_answer_returns_string(self, handler):
        context = "The NIST CSF has five core functions: Identify, Protect, Detect, Respond, Recover."
        answer = handler.generate_answer("What are the CSF functions?", context)
        assert isinstance(answer, str)
        assert len(answer) > 5

    def test_generate_answer_uses_context(self, handler):
        context = "Marshmallow is a fictional city in outer space with 42 moons."
        answer = handler.generate_answer("How many moons does Marshmallow have?", context)
        # The answer should reference 42 since that's in the context
        assert "42" in answer or "forty" in answer.lower()

    def test_empty_question_handled(self, handler):
        answer = handler.generate_answer("", "Some context")
        assert isinstance(answer, str)  # should not throw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
