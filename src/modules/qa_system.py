"""
Q&A System Module
==================
End-to-end question-answering pipeline for the AI File Assistant.

Flow:
  User question
      -> QueryProcessor   (semantic search -> top-k chunks)
      -> ContextBuilder   (format chunks -> context string)
      -> OpenAI GPT       (generate grounded answer)
      -> Return answer + sources + confidence

Returns structured result with answer, citations, and confidence score.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from src.modules.openai_handler import OpenAIHandler
from src.modules.query_processor import QueryProcessor
from src.modules.context_builder import ContextBuilder
from src.modules.vector_db_setup import VectorDB

logger = logging.getLogger(__name__)


# System prompt: instructs the LLM how to answer from context
_SYSTEM_PROMPT = """You are an expert AI assistant specializing in enterprise \
technology standards, cybersecurity frameworks, and digital governance. Your role \
is to provide comprehensive, well-explained answers based ONLY on the provided document context.

Guidelines:
- Explain concepts clearly: Do not just copy-paste from the text. Read the context and explain the answer in a structured, easy-to-understand manner suitable for a professional.
- Structure your response: Use clear headings, bullet points, or numbered lists to make the information highly organised and digestible.
- Strictly Grounded: Base your entire answer ONLY on the provided document context. Do not include outside knowledge.
- Contextualise: Briefly introduce the concept before diving into specifics to provide a better explanation.
- Always cite the source document and page number when referencing specific facts or claims (e.g., "[NIST.CSWP.29, pg 14]").
- If the context does not contain enough information, explicitly state so, but provide any partial information available.
- Be professional, precise, and highly analytical.
"""

_USER_PROMPT_TEMPLATE = """\
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Please answer the question using only the information from the context above. \
Cite the source documents when referencing specific information."""


class QASystem:
    """
    Complete Question-Answering system integrating all pipeline components.

    Attributes:
        vector_db       : VectorDB (ChromaDB or FAISS)
        openai_handler  : OpenAI API handler
        query_processor : Semantic search engine
        context_builder : Context assembler
        model           : GPT model to use for answer generation
        top_k           : Chunks to retrieve per query
        max_context_tokens: Token budget for context
    """

    def __init__(
        self,
        vector_db: VectorDB,
        openai_handler: Optional[OpenAIHandler] = None,
        model: Optional[str] = None,
        top_k: int = 5,
        max_context_tokens: int = 3000,
        min_search_score: float = 0.3
    ):
        """
        Initialize the Q&A system.

        Args:
            vector_db          : Configured VectorDB instance
            openai_handler     : Optional pre-configured API handler
            model              : GPT model (default: from .env OPENAI_MODEL)
            top_k              : Chunks to retrieve (5-10 recommended)
            max_context_tokens : Token budget (~3000 for GPT-3.5)
            min_search_score   : Minimum similarity to include a chunk
        """
        self.vector_db = vector_db
        self.openai_handler = openai_handler or OpenAIHandler()
        self.model = model  # None = use handler default from .env

        self.query_processor = QueryProcessor(
            vector_db=vector_db,
            openai_handler=self.openai_handler,
            default_top_k=top_k,
            min_score=min_search_score
        )
        self.context_builder = ContextBuilder(
            max_tokens=max_context_tokens,
            min_score=min_search_score,
            include_scores=True,
            deduplicate=True
        )

        # Session history and stats
        self.conversation_history: List[Dict[str, str]] = []
        self.session_stats = {
            "questions_answered": 0,
            "total_time": 0.0,
            "avg_confidence": 0.0,
            "total_sources_cited": 0
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        filter_source: Optional[str] = None,
        use_conversation_history: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question using the document knowledge base.

        Args:
            question               : Natural language question
            filter_source          : Restrict search to one PDF filename
            use_conversation_history: Include prior turns in the prompt

        Returns:
            dict with:
              - answer      (str)        : Generated answer text
              - sources     (List[dict]) : Cited sources with file/page/score
              - confidence  (float)      : Confidence score 0.0-1.0
              - search_results (List)    : Raw ranked chunks used
              - context     (str)        : Context block passed to LLM
              - model       (str)        : GPT model used
              - time_seconds(float)      : End-to-end latency
              - question    (str)        : Original question
              - error       (str|None)   : Error message if failed
        """
        result = {
            "question": question,
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "search_results": [],
            "context": "",
            "model": self.model or "default",
            "time_seconds": 0.0,
            "error": None
        }

        if not question or not question.strip():
            result["error"] = "Empty question"
            return result

        start = time.time()

        try:
            # Step 1: Semantic search
            logger.info(f"Q&A: '{question[:60]}'")
            search_results = self.query_processor.search_documents(
                question,
                filter_source=filter_source
            )
            result["search_results"] = search_results

            if not search_results:
                result["answer"] = (
                    "I could not find relevant information in the "
                    "document database to answer your question. "
                    "Please ensure PDFs have been ingested first."
                )
                result["error"] = "no_results"
                result["time_seconds"] = round(time.time() - start, 2)
                return result

            # Step 2: Build context
            ctx_meta = self.context_builder.build_context_with_metadata(
                search_results, question
            )
            context_str = ctx_meta["context"]
            result["context"] = context_str

            if not context_str:
                result["answer"] = (
                    "The retrieved document chunks did not meet the "
                    "relevance threshold to build a meaningful context. "
                    "Try rephrasing your question."
                )
                result["error"] = "low_relevance"
                result["time_seconds"] = round(time.time() - start, 2)
                return result

            # Step 3: Build full system message (our detailed instructions)
            system_msg = _SYSTEM_PROMPT

            # Step 4: For multi-turn, prepend recent conversation to context
            if use_conversation_history and self.conversation_history:
                prior = ""
                for i in range(0, len(self.conversation_history[-6:]), 2):
                    turns = self.conversation_history[-6:]
                    if i + 1 < len(turns):
                        prior += (
                            f"Previous Q: {turns[i]['content']}\n"
                            f"Previous A: {turns[i+1]['content']}\n\n"
                        )
                if prior:
                    context_str = prior + "Current Context:\n" + context_str

            # Step 5: Generate answer via existing generate_answer()
            model_override = self.model  # None = use handler's default
            answer_text = self.openai_handler.generate_answer(
                question=question,
                context=context_str,
                system_message=system_msg,
                temperature=0.3,
                max_tokens=600
            )

            # Step 5: Extract sources
            sources = self.context_builder.get_sources_summary(search_results)
            result["sources"] = sources

            # Step 6: Compute confidence
            confidence = self._compute_confidence(search_results, answer_text)

            # Populate result
            result["answer"] = answer_text
            result["confidence"] = confidence
            result["model"] = self.model or "default"
            result["time_seconds"] = round(time.time() - start, 2)

            # Update conversation history (for multi-turn)
            self.conversation_history.append(
                {"role": "user",    "content": question}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": answer_text}
            )

            # Update session stats
            self.session_stats["questions_answered"] += 1
            self.session_stats["total_time"] += result["time_seconds"]
            self.session_stats["total_sources_cited"] += len(sources)
            n = self.session_stats["questions_answered"]
            prev_avg = self.session_stats["avg_confidence"]
            self.session_stats["avg_confidence"] = round(
                (prev_avg * (n - 1) + confidence) / n, 3
            )

            logger.info(
                f"Answer generated in {result['time_seconds']}s | "
                f"confidence={confidence:.2f} | "
                f"sources={[s['source_file'] for s in sources]}"
            )

        except Exception as e:
            result["error"] = str(e)
            result["time_seconds"] = round(time.time() - start, 2)
            logger.error(f"Q&A failed: {e}")

        return result

    def answer_question_stream(
        self,
        question: str,
        filter_source: Optional[str] = None,
        use_conversation_history: bool = False
    ):
        """
        Stream the answer to a question using the document knowledge base.

        Yields:
            The first yielded item is a dict containing search results, sources, 
            confidence, time, etc. (setup meta-data).
            Subsequent yielded items are strings (the chunks of the answer).
            The final yielded item is a dict with the complete answer and updated stats.
        """
        result = {
            "question": question,
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "search_results": [],
            "context": "",
            "model": self.model or "default",
            "time_seconds": 0.0,
            "error": None
        }

        if not question or not question.strip():
            result["error"] = "Empty question"
            yield result
            return

        start = time.time()

        try:
            # Step 1: Semantic search
            logger.info(f"Q&A Stream: '{question[:60]}'")
            search_results = self.query_processor.search_documents(
                question,
                filter_source=filter_source
            )
            result["search_results"] = search_results

            if not search_results:
                result["answer"] = (
                    "I could not find relevant information in the "
                    "document database to answer your question. "
                    "Please ensure PDFs have been ingested first."
                )
                result["error"] = "no_results"
                result["time_seconds"] = round(time.time() - start, 2)
                yield result
                yield result["answer"]
                return

            # Step 2: Build context
            ctx_meta = self.context_builder.build_context_with_metadata(
                search_results, question
            )
            context_str = ctx_meta["context"]
            result["context"] = context_str

            if not context_str:
                result["answer"] = (
                    "The retrieved document chunks did not meet the "
                    "relevance threshold to build a meaningful context. "
                    "Try rephrasing your question."
                )
                result["error"] = "low_relevance"
                result["time_seconds"] = round(time.time() - start, 2)
                yield result
                yield result["answer"]
                return

            # Step 3: Build full system message
            system_msg = _SYSTEM_PROMPT

            # Step 4: Multi-turn context
            if use_conversation_history and self.conversation_history:
                prior = ""
                for i in range(0, len(self.conversation_history[-6:]), 2):
                    turns = self.conversation_history[-6:]
                    if i + 1 < len(turns):
                        prior += (
                            f"Previous Q: {turns[i]['content']}\n"
                            f"Previous A: {turns[i+1]['content']}\n\n"
                        )
                if prior:
                    context_str = prior + "Current Context:\n" + context_str

            # Step 5: Extract sources
            sources = self.context_builder.get_sources_summary(search_results)
            result["sources"] = sources

            # Calculate preliminary confidence (approximate based on search scores)
            result["confidence"] = self._compute_confidence(search_results, "")

            # Yield setup meta-data
            yield result

            # Yield chunks
            answer_text = ""
            for chunk in self.openai_handler.generate_answer_stream(
                question=question,
                context=context_str,
                system_message=system_msg,
                temperature=0.3,
                max_tokens=600
            ):
                answer_text += chunk
                yield chunk

            # Finalize
            confidence = self._compute_confidence(search_results, answer_text)
            result["answer"] = answer_text
            result["confidence"] = confidence
            result["time_seconds"] = round(time.time() - start, 2)

            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer_text})

            self.session_stats["questions_answered"] += 1
            self.session_stats["total_time"] += result["time_seconds"]
            self.session_stats["total_sources_cited"] += len(sources)
            n = self.session_stats["questions_answered"]
            prev_avg = self.session_stats["avg_confidence"]
            self.session_stats["avg_confidence"] = round(
                (prev_avg * (n - 1) + confidence) / n, 3
            )

            logger.info(
                f"Streamed answer generated in {result['time_seconds']}s | "
                f"confidence={confidence:.2f}"
            )

            # Yield the finalized result dict
            yield result

        except Exception as e:
            result["error"] = str(e)
            result["time_seconds"] = round(time.time() - start, 2)
            logger.error(f"Q&A Stream failed: {e}")
            yield result

    def answer_with_followup(
        self,
        question: str
    ) -> Dict[str, Any]:
        """
        Answer using conversation history (multi-turn Q&A).
        Automatically uses prior context for follow-up questions.
        """
        return self.answer_question(
            question, use_conversation_history=True
        )

    def answer_with_followup_stream(self, question: str):
        """
        Streaming Answer using conversation history.
        """
        return self.answer_question_stream(question, use_conversation_history=True)

    def reset_conversation(self):
        """Clear conversation history (start fresh session)."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_session_stats(self) -> Dict[str, Any]:
        """Return session statistics."""
        stats = self.session_stats.copy()
        n = stats["questions_answered"]
        stats["avg_time_seconds"] = round(
            stats["total_time"] / n, 2) if n else 0
        return stats

    def format_answer(self, qa_result: Dict[str, Any]) -> str:
        """
        Format a Q&A result into a human-readable string.

        Args:
            qa_result: Dict returned by answer_question()

        Returns:
            Formatted multi-line string for printing
        """
        lines = [
            "=" * 65,
            f"QUESTION: {qa_result['question']}",
            "=" * 65,
            "",
            qa_result.get("answer", "[No answer]"),
            "",
            "-" * 65,
            "SOURCES:",
        ]

        for src in qa_result.get("sources", []):
            pages_str = ", ".join(str(p) for p in src.get("pages", []))
            lines.append(
                f"  - {src['source_file']} "
                f"(pages: {pages_str}, "
                f"relevance: {src['avg_score']:.2f})"
            )

        conf = qa_result.get("confidence", 0)
        lines += [
            "",
            f"Confidence: {conf:.0%}  |  "
            f"Time: {qa_result.get('time_seconds', 0)}s  |  "
            f"Model: {qa_result.get('model', 'unknown')}",
            "=" * 65,
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        search_results: List[Dict],
        answer: str
    ) -> float:
        """
        Estimate answer confidence (0.0 – 1.0).

        Heuristic:
          - Start from average top-3 similarity scores
          - Penalise if the answer contains uncertainty phrases
          - Cap at 0.99
        """
        if not search_results:
            return 0.0

        top_scores = [r.get("score", 0) for r in search_results[:3]]
        base_confidence = sum(top_scores) / len(top_scores)

        # Penalise uncertainty phrases
        uncertainty_phrases = [
            "i don't know", "i cannot find", "not mentioned",
            "no information", "unclear", "not specified",
            "cannot determine", "insufficient"
        ]
        answer_lower = answer.lower()
        penalty = sum(0.1 for p in uncertainty_phrases if p in answer_lower)

        confidence = max(0.0, min(0.99, base_confidence - penalty))
        return round(confidence, 3)


# ------------------------------------------------------------------
# Module-level convenience function (matches PROMPT 14 spec)
# ------------------------------------------------------------------

def answer_question(
    question: str,
    vector_db: VectorDB,
    openai_handler: Optional[OpenAIHandler] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Answer a question using the document knowledge base.

    Args:
        question      : Natural language question
        vector_db     : Configured VectorDB instance
        openai_handler: Optional pre-built handler
        top_k         : Chunks to retrieve

    Returns:
        dict with answer, sources, confidence (see QASystem.answer_question)
    """
    system = QASystem(
        vector_db=vector_db,
        openai_handler=openai_handler,
        top_k=top_k
    )
    return system.answer_question(question)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from src.modules.chromadb_handler import ChromaDBHandler

    db = ChromaDBHandler()
    count = db.get_count()
    if count == 0:
        print("No data in ChromaDB. Run storage_pipeline.py first.")
        sys.exit(0)

    qa = QASystem(db)
    print(f"Q&A system ready ({count} chunks indexed)\n")

    demo_questions = [
        "What are the five core functions of the NIST Cybersecurity Framework?",
        "How should an organization handle a cybersecurity incident?",
        "What is the principle of least privilege?",
    ]

    for question in demo_questions:
        result = qa.answer_question(question)
        print(qa.format_answer(result))
        print()
