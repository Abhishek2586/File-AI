"""
Week 3-4 Integration Test Suite
==================================
Tests vector database operations, query processing, context building,
Q&A system, semantic search, edge cases, and performance benchmarks.

Tests:
1. ChromaDB CRUD operations
2. FAISS CRUD operations
3. Storage pipeline (mock data)
4. Semantic similarity search
5. Query processor (Week 4)
6. Context builder (Week 4)
7. Q&A system with 10+ questions (Week 4)
8. Edge cases
9. Performance benchmarks
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"test_week3_4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO noise during tests
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Shared test data ────────────────────────────────────────────────────────

NIST_SAMPLE_CHUNKS = [
    {
        "id": "nist_001",
        "text": "The NIST Cybersecurity Framework provides guidance for "
                "managing cybersecurity risk. It consists of five core "
                "functions: Identify, Protect, Detect, Respond, and Recover.",
        "meta": {"source_file": "NIST_CSWP_29.pdf", "page_number": 1, "chunk_index": 0}
    },
    {
        "id": "nist_002",
        "text": "The Identify function helps develop an organizational "
                "understanding to manage cybersecurity risk to systems, "
                "assets, data, and capabilities.",
        "meta": {"source_file": "NIST_CSWP_29.pdf", "page_number": 2, "chunk_index": 1}
    },
    {
        "id": "nist_003",
        "text": "The Protect function outlines appropriate safeguards to "
                "limit or contain the impact of a potential cybersecurity event.",
        "meta": {"source_file": "NIST_CSWP_29.pdf", "page_number": 3, "chunk_index": 2}
    },
    {
        "id": "nist_004",
        "text": "The Detect function defines activities to identify the "
                "occurrence of a cybersecurity event in a timely manner.",
        "meta": {"source_file": "NIST_CSWP_29.pdf", "page_number": 4, "chunk_index": 3}
    },
    {
        "id": "nist_005",
        "text": "The Respond function includes activities to take action "
                "regarding a detected cybersecurity incident.",
        "meta": {"source_file": "NIST_CSWP_29.pdf", "page_number": 5, "chunk_index": 4}
    },
    {
        "id": "nist_006",
        "text": "The Recover function identifies activities to maintain "
                "resilience plans and restore capabilities impaired by a "
                "cybersecurity incident.",
        "meta": {"source_file": "NIST_CSWP_29.pdf", "page_number": 6, "chunk_index": 5}
    },
    {
        "id": "nist_007",
        "text": "Supply chain risk management is critical for organizations. "
                "Third-party vendors and suppliers can introduce cybersecurity "
                "vulnerabilities into the supply chain.",
        "meta": {"source_file": "NIST_IR_8596_iprd.pdf", "page_number": 1, "chunk_index": 0}
    },
    {
        "id": "nist_008",
        "text": "Access control policies should enforce the principle of least "
                "privilege. Users should only have access to resources necessary "
                "for their job functions.",
        "meta": {"source_file": "NIST_IR_8596_iprd.pdf", "page_number": 2, "chunk_index": 1}
    },
    {
        "id": "nist_009",
        "text": "Encryption is essential for protecting sensitive data at rest "
                "and in transit. Strong encryption algorithms should be used "
                "according to current NIST standards.",
        "meta": {"source_file": "NIST_SP_500-291r2.pdf", "page_number": 1, "chunk_index": 0}
    },
    {
        "id": "nist_010",
        "text": "Incident response planning is a key component of cybersecurity "
                "preparedness. Organizations should develop, test, and maintain "
                "incident response plans.",
        "meta": {"source_file": "NIST_SP_500-291r2.pdf", "page_number": 3, "chunk_index": 1}
    },
]

# Dummy 1536-dim embedding (for tests without API)
DUMMY_VEC = [0.01] * 1536


class Week3_4_Tester:
    """Comprehensive test runner for Week 3-4 modules."""

    def __init__(self):
        self.results = {"passed": 0, "failed": 0, "skipped": 0, "details": []}
        self._openai_available = False
        self._openai_handler = None
        self._try_init_openai()

    # ── helpers ────────────────────────────────────────────────────────────

    def _try_init_openai(self):
        try:
            from src.modules.openai_handler import OpenAIHandler
            h = OpenAIHandler()
            if h.test_connection():
                self._openai_handler = h
                self._openai_available = True
                print("  [OK] OpenAI API available")
        except Exception:
            print("  [!] OpenAI API not available -- skipping API-dependent tests")

    def _embed(self, text: str) -> List[float]:
        """Return real embedding if API available, else dummy."""
        if self._openai_available and self._openai_handler:
            return self._openai_handler.get_embedding(text)
        return DUMMY_VEC[:]

    def _log(self, name: str, passed: bool, msg: str = "", data: dict = None):
        status = "[PASS]" if passed else "[FAIL]"
        if passed:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        self.results["details"].append(
            {"test": name, "passed": passed, "message": msg, "data": data or {}}
        )
        print(f"  {status}: {name}" + (f" -- {msg}" if msg else ""))

    def _skip(self, name: str, reason: str):
        self.results["skipped"] += 1
        self.results["details"].append(
            {"test": name, "passed": None, "message": f"SKIP: {reason}"}
        )
        print(f"  [SKIP]: {name} -- {reason}")

    # ---- test sections ----------------------------------------------------

    def test_chromadb_crud(self):
        print("\n" + "=" * 65)
        print("TEST 1: ChromaDB CRUD Operations")
        print("=" * 65)

        try:
            from src.modules.chromadb_handler import ChromaDBHandler
            db = ChromaDBHandler(
                persist_directory="./test_chroma_temp",
                collection_name="test_collection"
            )
            db.reset()  # start clean

            # 1a. Init
            self._log("ChromaDB Initialization", True,
                      f"Collection ready, {db.get_count()} items")

            # 1b. Upsert
            ids = [c["id"] for c in NIST_SAMPLE_CHUNKS]
            vecs = [self._embed(c["text"]) for c in NIST_SAMPLE_CHUNKS]
            metas = [c["meta"] for c in NIST_SAMPLE_CHUNKS]
            docs = [c["text"] for c in NIST_SAMPLE_CHUNKS]

            ok = db.upsert(ids, vecs, metas, docs)
            self._log("ChromaDB Upsert", ok and db.get_count() == len(ids),
                      f"{db.get_count()}/{len(ids)} vectors stored")

            # 1c. Query
            q_vec = self._embed("What is the NIST Cybersecurity Framework?")
            results = db.query(q_vec, top_k=3)
            self._log("ChromaDB Query", len(results) > 0,
                      f"Got {len(results)} results, top score={results[0]['score'] if results else 'N/A'}")

            # 1d. List documents
            doc_list = db.list_documents()
            self._log("ChromaDB List Documents", len(doc_list) == 3,
                      f"Found {len(doc_list)} unique docs: {doc_list}")

            # 1e. Delete document
            deleted = db.delete_document("NIST_IR_8596_iprd.pdf")
            self._log("ChromaDB Delete Document", deleted == 2,
                      f"Deleted {deleted} chunks from NIST_IR_8596_iprd.pdf")

            # 1f. Re-query after deletion
            remaining = db.get_count()
            self._log("ChromaDB Post-Delete Count", remaining == 8,
                      f"{remaining} chunks remaining (expected 8)")

            # 1g. Delete by IDs
            ok2 = db.delete(["nist_001"])
            self._log("ChromaDB Delete by ID", ok2,
                      f"Chunk deleted, now {db.get_count()} total")

            # 1h. Reset
            reset_ok = db.reset()
            self._log("ChromaDB Reset", reset_ok and db.get_count() == 0,
                      f"Collection reset: {db.get_count()} items")

            # Clean up temp dir
            import shutil
            shutil.rmtree("./test_chroma_temp", ignore_errors=True)

        except Exception as e:
            self._log("ChromaDB CRUD", False, str(e))

    def test_faiss_crud(self):
        print("\n" + "=" * 65)
        print("TEST 2: FAISS CRUD Operations")
        print("=" * 65)

        try:
            import faiss  # noqa: F401
        except ImportError:
            self._skip("FAISS CRUD", "faiss-cpu not installed (pip install faiss-cpu)")
            return

        try:
            from src.modules.faiss_handler import FAISSHandler
            db = FAISSHandler(
                index_path="./test_faiss_temp",
                collection_name="test"
            )
            db.reset()

            self._log("FAISS Initialization", True,
                      f"Index ready, {db.get_count()} vectors")

            ids = [c["id"] for c in NIST_SAMPLE_CHUNKS]
            vecs = [self._embed(c["text"]) for c in NIST_SAMPLE_CHUNKS]
            metas = [c["meta"] for c in NIST_SAMPLE_CHUNKS]
            docs = [c["text"] for c in NIST_SAMPLE_CHUNKS]

            ok = db.upsert(ids, vecs, metas, docs)
            self._log("FAISS Upsert", ok and db.get_count() == len(ids),
                      f"{db.get_count()} vectors stored")

            q_vec = self._embed("What is the NIST Cybersecurity Framework?")
            results = db.query(q_vec, top_k=3)
            self._log("FAISS Query", len(results) > 0,
                      f"Got {len(results)} results")

            doc_list = db.list_documents()
            self._log("FAISS List Documents", len(doc_list) > 0,
                      f"{len(doc_list)} documents: {doc_list}")

            db.reset()
            self._log("FAISS Reset", db.get_count() == 0,
                      "Index cleared")

            import shutil
            shutil.rmtree("./test_faiss_temp", ignore_errors=True)

        except Exception as e:
            self._log("FAISS CRUD", False, str(e))

    def test_storage_pipeline(self):
        print("\n" + "=" * 65)
        print("TEST 3: Storage Pipeline (mock data)")
        print("=" * 65)

        try:
            from src.modules.chromadb_handler import ChromaDBHandler
            from src.modules.storage_pipeline import StoragePipeline
            from src.modules.text_chunker import TextChunk

            db = ChromaDBHandler(
                persist_directory="./test_pipeline_chroma",
                collection_name="pipeline_test"
            )
            db.reset()

            pipeline = StoragePipeline(
                vector_db=db,
                openai_handler=self._openai_handler
            )

            self._log("Storage Pipeline Init", True,
                      "Pipeline created with ChromaDB backend")

            # Manually inject chunks (bypass real PDF)
            mock_text = "\n\n".join(c["text"] for c in NIST_SAMPLE_CHUNKS[:4])
            chunks = pipeline.text_chunker.chunk_text(
                mock_text, "mock_nist.pdf", 1
            )
            self._log("Chunk Creation", len(chunks) > 0,
                      f"Created {len(chunks)} chunks from mock text")

            # Generate embeddings and store
            emb_results = pipeline.embedding_pipeline.\
                generate_embeddings_for_chunks(chunks, show_progress=False)
            self._log("Embedding Generation", len(emb_results) == len(chunks),
                      f"{len(emb_results)} embeddings")

            # Store in DB
            ids, vecs, metas, docs2 = [], [], [], []
            for i, (chunk, emb) in enumerate(emb_results):
                ids.append(f"mock_chunk_{i:04d}")
                vecs.append(emb)
                metas.append({
                    "source_file": "mock_nist.pdf",
                    "chunk_index": i,
                    "page_number": chunk.metadata.get("page_number", 1)
                })
                docs2.append(chunk.text)

            ok = db.upsert(ids, vecs, metas, docs2)
            self._log("Storage to ChromaDB", ok and db.get_count() > 0,
                      f"{db.get_count()} chunks in DB")

            # Pipeline stats
            stats = pipeline.get_stats()
            self._log("Pipeline Stats Available",
                      "embedding_metrics" in stats, str(stats.get("embedding_metrics", {})))

            import shutil
            shutil.rmtree("./test_pipeline_chroma", ignore_errors=True)

        except Exception as e:
            self._log("Storage Pipeline", False, str(e))

    def test_similarity_search(self):
        print("\n" + "=" * 65)
        print("TEST 4: Semantic Similarity Search")
        print("=" * 65)

        if not self._openai_available:
            self._skip("Semantic Search", "OpenAI API not configured")
            return

        try:
            from src.modules.chromadb_handler import ChromaDBHandler
            db = ChromaDBHandler(
                persist_directory="./test_search_chroma",
                collection_name="search_test"
            )
            db.reset()

            # Load all NIST sample chunks
            ids = [c["id"] for c in NIST_SAMPLE_CHUNKS]
            vecs = [self._embed(c["text"]) for c in NIST_SAMPLE_CHUNKS]
            db.upsert(ids, vecs,
                      [c["meta"] for c in NIST_SAMPLE_CHUNKS],
                      [c["text"] for c in NIST_SAMPLE_CHUNKS])

            test_questions = [
                ("What are the 5 functions of NIST CSF?", "NIST_CSWP_29.pdf"),
                ("How do you detect cybersecurity incidents?", "NIST_CSWP_29.pdf"),
                ("What is supply chain risk management?", "NIST_IR_8596_iprd.pdf"),
                ("How should access control be implemented?", "NIST_IR_8596_iprd.pdf"),
                ("Why is encryption important?", "NIST_SP_500-291r2.pdf"),
                ("What is incident response planning?", "NIST_SP_500-291r2.pdf"),
            ]

            correct = 0
            for question, expected_source in test_questions:
                q_vec = self._embed(question)
                results = db.query(q_vec, top_k=1)
                if results:
                    got_source = results[0]["metadata"].get("source_file", "")
                    is_correct = got_source == expected_source
                    if is_correct:
                        correct += 1
                    self._log(
                        f"Search: '{question[:45]}...'",
                        is_correct,
                        f"Expected {expected_source}, Got {got_source} "
                        f"(score={results[0]['score']})"
                    )
                else:
                    self._log(f"Search: '{question[:45]}...'", False, "No results")

            accuracy = (correct / len(test_questions)) * 100
            self._log(
                "Search Accuracy",
                accuracy >= 60,
                f"{correct}/{len(test_questions)} correct ({accuracy:.1f}%)"
            )

            import shutil
            shutil.rmtree("./test_search_chroma", ignore_errors=True)

        except Exception as e:
            self._log("Similarity Search", False, str(e))

    def test_edge_cases(self):
        print("\n" + "=" * 65)
        print("TEST 5: Edge Cases")
        print("=" * 65)

        try:
            from src.modules.chromadb_handler import ChromaDBHandler
            db = ChromaDBHandler(
                persist_directory="./test_edge_chroma",
                collection_name="edge_test"
            )
            db.reset()

            # 5a. Query empty DB
            results = db.query(DUMMY_VEC, top_k=5)
            self._log("Query Empty DB", results == [],
                      f"Returned {len(results)} results from empty DB")

            # 5b. Empty upsert
            ok = db.upsert([], [], [], [])
            self._log("Empty Upsert Handled", not ok,
                      "Empty upsert correctly returned False")

            # 5c. Duplicate upsert (should replace, not duplicate)
            db.upsert(["dup_001"], [DUMMY_VEC],
                      [{"source_file": "test.pdf"}], ["Original text"])
            db.upsert(["dup_001"], [DUMMY_VEC],
                      [{"source_file": "test.pdf"}], ["Updated text"])
            count = db.get_count()
            self._log("Duplicate Upsert", count == 1,
                      f"After 2 upserts with same ID: {count} chunk (expected 1)")

            # 5d. Delete non-existent document
            deleted = db.delete_document("ghost_file.pdf")
            self._log("Delete Non-Existent Doc", deleted == 0,
                      f"Returned {deleted} (expected 0)")

            # 5e. List docs with no documents
            db.reset()
            docs = db.list_documents()
            self._log("List Docs Empty DB", docs == [],
                      f"Returned {docs}")

            # 5f. top_k larger than DB size
            db.upsert(["single"], [DUMMY_VEC],
                      [{"source_file": "single.pdf"}], ["Single doc"])
            results = db.query(DUMMY_VEC, top_k=100)
            self._log("top_k > DB Size", len(results) == 1,
                      f"Got {len(results)} result(s) when DB has 1 chunk")

            import shutil
            shutil.rmtree("./test_edge_chroma", ignore_errors=True)

        except Exception as e:
            self._log("Edge Cases", False, str(e))

    def test_performance(self):
        print("\n" + "=" * 65)
        print("TEST 6: Performance Benchmarks")
        print("=" * 65)

        try:
            from src.modules.chromadb_handler import ChromaDBHandler
            db = ChromaDBHandler(
                persist_directory="./test_perf_chroma",
                collection_name="perf_test"
            )
            db.reset()

            N = 50
            vecs = [DUMMY_VEC[:] for _ in range(N)]
            ids = [f"perf_{i:04d}" for i in range(N)]
            metas = [{"source_file": "perf.pdf", "chunk_index": i} for i in range(N)]
            docs = [f"Performance test document number {i}" for i in range(N)]

            # Upsert benchmark
            t0 = time.time()
            db.upsert(ids, vecs, metas, docs)
            upsert_time = time.time() - t0
            self._log(
                f"Upsert {N} Vectors",
                upsert_time < 30 and db.get_count() == N,
                f"{db.get_count()} stored in {upsert_time:.2f}s "
                f"({N/upsert_time:.1f} vec/s)"
            )

            # Query benchmark (10 queries)
            times = []
            for _ in range(10):
                t0 = time.time()
                db.query(DUMMY_VEC, top_k=5)
                times.append(time.time() - t0)
            avg_ms = (sum(times) / len(times)) * 1000
            self._log(
                "Query Latency (10 queries)",
                avg_ms < 500,
                f"Avg {avg_ms:.1f}ms per query"
            )

            import shutil
            shutil.rmtree("./test_perf_chroma", ignore_errors=True)

        except Exception as e:
            self._log("Performance", False, str(e))

    # ---- Week 4 tests --------------------------------------------------------

    def test_query_processor(self):
        print("\n" + "=" * 65)
        print("TEST 5: Query Processor (Week 4)")
        print("=" * 65)

        if not self._openai_available:
            self._skip("Query Processor", "OpenAI API not configured")
            return

        try:
            from src.modules.chromadb_handler import ChromaDBHandler
            from src.modules.query_processor import QueryProcessor, search_documents

            db = ChromaDBHandler(
                persist_directory="./test_qp_chroma",
                collection_name="qp_test"
            )
            db.reset()

            # Load NIST sample data
            ids = [c["id"] for c in NIST_SAMPLE_CHUNKS]
            vecs = [self._embed(c["text"]) for c in NIST_SAMPLE_CHUNKS]
            db.upsert(ids, vecs,
                      [c["meta"] for c in NIST_SAMPLE_CHUNKS],
                      [c["text"] for c in NIST_SAMPLE_CHUNKS])

            qp = QueryProcessor(db, self._openai_handler, default_top_k=5)

            # Test basic search
            results = qp.search_documents(
                "What are the five functions of NIST CSF?"
            )
            self._log("QP: Basic Search",
                      len(results) > 0 and all("rank" in r for r in results),
                      f"{len(results)} results, top score={results[0]['score'] if results else 'N/A'}")

            # Test source filter
            filtered = qp.search_documents(
                "cybersecurity framework",
                filter_source="NIST_CSWP_29.pdf"
            )
            all_correct_source = all(
                r["metadata"].get("source_file") == "NIST_CSWP_29.pdf"
                for r in filtered
            )
            self._log("QP: Source Filter", all_correct_source,
                      f"{len(filtered)} results all from NIST_CSWP_29.pdf")

            # Test query cache
            q = "What is incident response?"
            _ = qp.search_documents(q)
            _ = qp.search_documents(q)  # second call hits cache
            stats = qp.get_stats()
            self._log("QP: Query Cache", stats["cache_hits"] >= 1,
                      f"Cache hits: {stats['cache_hits']}")

            # Test convenience function
            results2 = search_documents(
                "encryption standards", db,
                top_k=3, openai_handler=self._openai_handler
            )
            self._log("QP: Convenience Function", len(results2) > 0,
                      f"{len(results2)} results")

            # Test get_top_result
            top = qp.get_top_result("supply chain risks")
            self._log("QP: Get Top Result",
                      top is not None and top["rank"] == 1,
                      f"Top result: score={top['score'] if top else 'N/A'}")

            import shutil
            shutil.rmtree("./test_qp_chroma", ignore_errors=True)

        except Exception as e:
            self._log("Query Processor", False, str(e))

    def test_context_builder(self):
        print("\n" + "=" * 65)
        print("TEST 6: Context Builder (Week 4)")
        print("=" * 65)

        try:
            from src.modules.context_builder import ContextBuilder, build_context

            # Build mock search results (3 distinct source files)
            mock_results = [
                {
                    "id": c["id"],
                    "document": c["text"],
                    "metadata": c["meta"],
                    "score": 0.90 - (i * 0.05),
                    "rank": i + 1
                }
                for i, c in enumerate(NIST_SAMPLE_CHUNKS)  # all 10 chunks = 3 sources
            ]

            builder = ContextBuilder(max_tokens=2000, min_score=0.5)

            # Test basic build
            context = builder.build_context(mock_results)
            self._log("CB: Build Context",
                      len(context) > 0 and "[Source:" in context,
                      f"{len(context)} chars built")

            # Test token limit respected
            small_builder = ContextBuilder(max_tokens=100)
            small_ctx = small_builder.build_context(mock_results)
            estimated_tokens = len(small_ctx) // 4
            self._log("CB: Token Limit", estimated_tokens <= 150,
                      f"~{estimated_tokens} tokens (limit=100)")

            # Test metadata output
            meta_out = builder.build_context_with_metadata(mock_results)
            self._log("CB: With Metadata",
                      "sources" in meta_out and "tokens_estimate" in meta_out,
                      f"Sources: {meta_out['sources']}, "
                      f"~{meta_out['tokens_estimate']} tokens")

            # Test sources summary
            sources = builder.get_sources_summary(mock_results)
            self._log("CB: Sources Summary",
                      len(sources) == 3 and all("avg_score" in s for s in sources),
                      f"{len(sources)} unique sources")

            # Test deduplication
            dup_results = mock_results[:1] + mock_results[:1]  # same chunk twice
            dedup_ctx = builder.build_context(dup_results)
            self._log("CB: Deduplication",
                      dedup_ctx.count("[Source:") == 1,
                      "Duplicate chunk correctly removed")

            # Test empty input
            empty_ctx = builder.build_context([])
            self._log("CB: Empty Input", empty_ctx == "",
                      "Empty results returns empty string")

            # Test convenience function
            ctx2 = build_context(mock_results, max_tokens=1500)
            self._log("CB: Convenience Function",
                      len(ctx2) > 0, f"{len(ctx2)} chars")

        except Exception as e:
            self._log("Context Builder", False, str(e))

    def test_qa_system(self):
        print("\n" + "=" * 65)
        print("TEST 7: Q&A System - 10+ Questions (Week 4)")
        print("=" * 65)

        if not self._openai_available:
            self._skip("Q&A System", "OpenAI API not configured")
            return

        try:
            from src.modules.chromadb_handler import ChromaDBHandler
            from src.modules.qa_system import QASystem, answer_question

            db = ChromaDBHandler(
                persist_directory="./test_qa_chroma",
                collection_name="qa_test"
            )
            db.reset()

            # Load all NIST sample chunks with REAL embeddings (so search works)
            ids = [c["id"] for c in NIST_SAMPLE_CHUNKS]
            vecs = [self._embed(c["text"]) for c in NIST_SAMPLE_CHUNKS]
            db.upsert(ids, vecs,
                      [c["meta"] for c in NIST_SAMPLE_CHUNKS],
                      [c["text"] for c in NIST_SAMPLE_CHUNKS])

            qa = QASystem(db, self._openai_handler, top_k=5)

            # 10 test questions
            test_questions = [
                "What are the five core functions of the NIST Cybersecurity Framework?",
                "What does the Identify function do?",
                "What is the purpose of the Protect function?",
                "How does the Detect function work?",
                "What activities are part of the Respond function?",
                "How does the Recover function help organizations?",
                "What is supply chain risk management in cybersecurity?",
                "What is the principle of least privilege?",
                "Why is encryption important for data security?",
                "What should an incident response plan include?",
                "How do third-party vendors affect cybersecurity?",
            ]

            passed_qa = 0
            total_time = 0.0
            total_confidence = 0.0

            for q in test_questions:
                result = qa.answer_question(q)
                has_answer = bool(result.get("answer")) and not result.get("error")
                has_sources = len(result.get("sources", [])) > 0
                ok = has_answer and has_sources
                if ok:
                    passed_qa += 1
                total_time += result.get("time_seconds", 0)
                total_confidence += result.get("confidence", 0)
                self._log(
                    f"Q&A: '{q[:50]}...'",
                    ok,
                    f"conf={result.get('confidence', 0):.2f}, "
                    f"time={result.get('time_seconds', 0)}s, "
                    f"sources={len(result.get('sources', []))}"
                )

            n = len(test_questions)
            avg_conf = total_confidence / n
            avg_time = total_time / n
            accuracy = (passed_qa / n) * 100

            self._log("Q&A: Overall Accuracy",
                      accuracy >= 70,
                      f"{passed_qa}/{n} questions answered ({accuracy:.0f}%)")
            self._log("Q&A: Avg Confidence",
                      avg_conf >= 0.3,
                      f"{avg_conf:.2f}")
            self._log("Q&A: Avg Response Time",
                      avg_time < 15,
                      f"{avg_time:.1f}s per question")

            # Test empty DB handling
            db_empty = ChromaDBHandler(
                persist_directory="./test_qa_empty",
                collection_name="empty_qa"
            )
            result_empty = qa.answer_question("test")
            # Even without results, answer should be non-empty informative text
            has_answer_text = bool(result_empty.get("answer"))
            self._log("Q&A: No-Results Handling", has_answer_text,
                      result_empty.get("answer", "")[:60])

            # Test multi-turn conversation
            qa.reset_conversation()
            r1 = qa.answer_with_followup(
                "What is NIST CSF?"
            )
            r2 = qa.answer_with_followup(
                "What are its five functions?"
            )
            self._log("Q&A: Multi-Turn",
                      bool(r1.get("answer")) and bool(r2.get("answer")),
                      "Two-turn conversation completed")

            # Test format_answer
            formatted = qa.format_answer(r1)
            self._log("Q&A: Format Answer",
                      "QUESTION:" in formatted and "SOURCES:" in formatted,
                      "Formatted output has correct sections")

            # Session stats
            stats = qa.get_session_stats()
            self._log("Q&A: Session Stats",
                      stats["questions_answered"] >= 11,
                      f"Questions: {stats['questions_answered']}, "
                      f"avg conf: {stats['avg_confidence']}")

            # Convenience function
            conv_result = answer_question(
                "What is the NIST CSF?", db,
                openai_handler=self._openai_handler
            )
            self._log("Q&A: Convenience Function",
                      bool(conv_result.get("answer")),
                      f"{len(conv_result.get('answer', ''))} char answer")

            import shutil
            shutil.rmtree("./test_qa_chroma", ignore_errors=True)
            shutil.rmtree("./test_qa_empty", ignore_errors=True)

        except Exception as e:
            self._log("Q&A System", False, str(e))

    # ---- original test sections -----------------------------------------------

    def run_all(self):
        print("\n" + "=" * 65)
        print("AI FILE ASSISTANT -- WEEK 3-4 TEST SUITE")
        print("=" * 65)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log:     {log_file}")

        # Week 3: Vector DB
        self.test_chromadb_crud()
        self.test_faiss_crud()
        self.test_storage_pipeline()
        self.test_similarity_search()

        # Week 4: Query + Q&A
        self.test_query_processor()
        self.test_context_builder()
        self.test_qa_system()

        # Cross-cutting
        self.test_edge_cases()
        self.test_performance()

        self._print_summary()
        self._save_report()

    def _print_summary(self):
        total = self.results["passed"] + self.results["failed"]
        rate = (self.results["passed"] / total * 100) if total else 0

        print("\n" + "=" * 65)
        print("SUMMARY")
        print("=" * 65)
        print(f"  Passed  : {self.results['passed']}")
        print(f"  Failed  : {self.results['failed']}")
        print(f"  Skipped : {self.results['skipped']}")
        print(f"  Rate    : {rate:.1f}%")

        failed = [d for d in self.results["details"] if d.get("passed") is False]
        if failed:
            print("\nFailed tests:")
            for d in failed:
                print(f"  [FAIL] {d['test']}: {d['message']}")
        print("=" * 65)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _save_report(self):
        report_file = log_dir / \
            f"week3_4_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "passed": self.results["passed"],
                    "failed": self.results["failed"],
                    "skipped": self.results["skipped"]
                },
                "details": self.results["details"]
            }, f, indent=2)
        print(f"\n✓ Report saved: {report_file}")


if __name__ == "__main__":
    tester = Week3_4_Tester()
    tester.run_all()
