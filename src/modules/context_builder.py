"""
Context Builder Module
=======================
Assembles retrieved document chunks into coherent LLM-ready context.

Handles:
- Source-attributed formatting (NIST document + page)
- Token-limit awareness (smart truncation)
- Deduplication of near-identical chunks
- Multiple output formats (plain text, structured dict)
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ~= 4 characters (OpenAI standard)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to stay within a token budget."""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [...]"


class ContextBuilder:
    """
    Builds structured context strings from similarity-search results.

    The context is designed to be inserted into an LLM prompt between
    a system instruction and the user question.

    Example output format:
        [Source: NIST_CSWP_29.pdf | Page 1 | Relevance: 0.92]
        The NIST Cybersecurity Framework provides guidance for managing...

        [Source: NIST_IR_8596_iprd.pdf | Page 2 | Relevance: 0.87]
        Supply chain risk management is critical for organizations...

    Attributes:
        max_tokens       : Hard token ceiling for the assembled context
        max_chunks       : Max number of chunks to include
        min_score        : Minimum similarity score to include a chunk
        include_scores   : Whether to show relevance scores in output
        deduplicate      : Whether to drop near-identical chunks
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        max_chunks: int = 8,
        min_score: float = 0.3,
        include_scores: bool = True,
        deduplicate: bool = True
    ):
        """
        Initialize the context builder.

        Args:
            max_tokens   : Token budget for the full context block
            max_chunks   : Maximum chunks to incorporate
            min_score    : Drop chunks below this similarity score
            include_scores: Show [Relevance: X.XX] in output
            deduplicate  : Skip chunks mostly identical to already-added ones
        """
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        self.min_score = min_score
        self.include_scores = include_scores
        self.deduplicate = deduplicate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(
        self,
        search_results: List[Dict[str, Any]],
        question: Optional[str] = None
    ) -> str:
        """
        Build a formatted context string from search results.

        Args:
            search_results : List of result dicts from QueryProcessor
            question       : Optional question (used only for logging)

        Returns:
            Formatted context string ready to insert into LLM prompt.
            Returns empty string if no valid chunks found.
        """
        if not search_results:
            logger.warning("build_context called with empty results")
            return ""

        # --- Filter by min_score ---
        valid = [r for r in search_results
                 if r.get("score", 0) >= self.min_score]

        if not valid:
            logger.warning(
                f"All {len(search_results)} results below min_score "
                f"({self.min_score})"
            )
            return ""

        # --- Sort by score descending (already sorted from DB, but ensure) ---
        valid = sorted(valid, key=lambda r: r.get("score", 0), reverse=True)

        # --- Deduplicate very similar chunks ---
        if self.deduplicate:
            valid = self._deduplicate(valid)

        # --- Respect max_chunks limit ---
        valid = valid[:self.max_chunks]

        # --- Build formatted blocks within token budget ---
        context_parts = []
        tokens_used = 0
        token_budget = self.max_tokens

        for result in valid:
            block = self._format_chunk(result)
            block_tokens = _estimate_tokens(block)

            if tokens_used + block_tokens > token_budget:
                # Try to fit a truncated version
                remaining = token_budget - tokens_used
                if remaining < 50:  # < 200 chars, not worth it
                    break
                block = _truncate_to_tokens(block, remaining)
                context_parts.append(block)
                tokens_used += _estimate_tokens(block)
                break

            context_parts.append(block)
            tokens_used += block_tokens

        context = "\n\n".join(context_parts)

        logger.info(
            f"Context built: {len(context_parts)} chunks, "
            f"~{tokens_used} tokens"
            + (f" for: '{question[:50]}'" if question else "")
        )
        return context

    def build_context_with_metadata(
        self,
        search_results: List[Dict[str, Any]],
        question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context and return enriched metadata alongside it.

        Returns:
            dict with:
              - context        : Formatted context string
              - chunks_used    : Number of chunks included
              - tokens_estimate: Estimated token count
              - sources        : List of unique source files used
              - scores         : List of similarity scores used
              - truncated      : Whether any chunks were truncated
        """
        context = self.build_context(search_results, question)

        used = [r for r in search_results
                if r.get("score", 0) >= self.min_score][:self.max_chunks]

        sources = list(dict.fromkeys(
            r["metadata"].get("source_file", "unknown")
            for r in used if r.get("metadata")
        ))
        scores = [r.get("score", 0) for r in used]

        return {
            "context": context,
            "chunks_used": len(used),
            "tokens_estimate": _estimate_tokens(context),
            "sources": sources,
            "scores": scores,
            "truncated": "[...]" in context
        }

    def get_sources_summary(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract a deduplicated list of source references.

        Returns:
            List of dicts with source_file, pages, avg_score
        """
        sources: Dict[str, Dict] = {}
        for r in search_results:
            meta = r.get("metadata", {})
            src = meta.get("source_file", "unknown")
            pg = meta.get("page_number", 0)
            score = r.get("score", 0)

            if src not in sources:
                sources[src] = {"source_file": src, "pages": set(),
                                "scores": []}
            sources[src]["pages"].add(pg)
            sources[src]["scores"].append(score)

        result = []
        for src, data in sources.items():
            result.append({
                "source_file": src,
                "pages": sorted(data["pages"]),
                "avg_score": round(
                    sum(data["scores"]) / len(data["scores"]), 3
                )
            })

        return sorted(result, key=lambda x: x["avg_score"], reverse=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_chunk(self, result: Dict[str, Any]) -> str:
        """Format a single chunk into a labelled text block."""
        meta = result.get("metadata", {})
        source = meta.get("source_file", "Unknown Document")
        page = meta.get("page_number", "?")
        score = result.get("score", 0)
        text = result.get("document", "")

        if self.include_scores:
            header = (
                f"[Source: {source} | Page {page} | "
                f"Relevance: {score:.2f}]"
            )
        else:
            header = f"[Source: {source} | Page {page}]"

        return f"{header}\n{text.strip()}"

    def _deduplicate(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove chunks with >80% word overlap against already-kept chunks.
        Uses Jaccard similarity on word sets.
        """
        kept = []
        kept_words: List[set] = []

        for r in results:
            doc = r.get("document", "")
            words = set(re.findall(r"\w+", doc.lower()))

            duplicate = False
            for existing_words in kept_words:
                if not existing_words:
                    continue
                union = existing_words | words
                intersection = existing_words & words
                jaccard = len(intersection) / len(union) if union else 0
                if jaccard >= 0.8:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(r)
                kept_words.append(words)

        if len(kept) < len(results):
            logger.debug(
                f"Deduplicated {len(results) - len(kept)} near-duplicate chunks"
            )
        return kept


# ------------------------------------------------------------------
# Module-level convenience function (matches PROMPT 13 spec)
# ------------------------------------------------------------------

def build_context(
    search_results: List[Dict[str, Any]],
    max_tokens: int = 3000
) -> str:
    """
    Build context from search results within a token limit.

    Args:
        search_results : Results from QueryProcessor.search_documents()
        max_tokens     : Maximum token budget (~3000 recommended for GPT-3.5)

    Returns:
        Formatted context string for LLM prompt insertion
    """
    builder = ContextBuilder(max_tokens=max_tokens)
    return builder.build_context(search_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo with mock search results
    mock_results = [
        {
            "id": "chunk_001",
            "document": "The NIST Cybersecurity Framework provides guidance "
                        "for managing cybersecurity risk through five core "
                        "functions: Identify, Protect, Detect, Respond, Recover.",
            "metadata": {"source_file": "NIST_CSWP_29.pdf", "page_number": 1},
            "score": 0.92,
            "rank": 1
        },
        {
            "id": "chunk_002",
            "document": "The Protect function outlines appropriate safeguards "
                        "to ensure delivery of critical infrastructure services.",
            "metadata": {"source_file": "NIST_CSWP_29.pdf", "page_number": 3},
            "score": 0.85,
            "rank": 2
        },
        {
            "id": "chunk_003",
            "document": "Supply chain risk management requires identifying "
                        "and assessing risks introduced by third-party vendors.",
            "metadata": {"source_file": "NIST_IR_8596_iprd.pdf", "page_number": 2},
            "score": 0.78,
            "rank": 3
        },
    ]

    builder = ContextBuilder(max_tokens=2000)
    context = builder.build_context(
        mock_results, question="What is the NIST CSF?"
    )

    print("=" * 60)
    print("BUILT CONTEXT:")
    print("=" * 60)
    print(context)
    print("\n" + "=" * 60)
    sources = builder.get_sources_summary(mock_results)
    print("SOURCES:")
    for s in sources:
        print(f"  {s['source_file']}: pages {s['pages']}, "
              f"avg score {s['avg_score']}")
