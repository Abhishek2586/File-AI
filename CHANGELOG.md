# CHANGELOG

All notable changes to the AI File Assistant project are documented here.

---

## [Week 8] — 2026-02-27

### Added
- **Dockerfile** for containerized deployment
- **docker-compose.yml** for single-command startup
- **GitHub Actions CI/CD** pipeline (`.github/workflows/ci.yml`)
- **Unit Tests**: test_pdf_processor.py, test_text_cleaner.py, test_chunker.py, test_openai_handler.py, test_vector_db.py, test_qa_system.py
- **Integration Tests**: test_integration.py covering end-to-end pipeline
- **Documentation** in `docs/`:
  - `ARCHITECTURE.md` — module diagrams and data flows
  - `UAT_Test_Scenarios.md` — 20 user acceptance test scenarios
  - `SECURITY_CHECKLIST.md` — security review checklist
  - `USER_GUIDE.md` — end-user facing guide with FAQ

---

## [Week 7] — 2026-02-26

### Fixed
- `StoragePipeline.process_pdf()` called with invalid `file_path` keyword → corrected to `pdf_path`
- `StoragePipeline.process_pdf()` received unsupported kwargs `chunk_size`, `chunk_overlap` → removed

---

## [Week 6] — 2026-02-26

### Added
- **Settings Page** (`ui/settings_page.py`): Model selection, temperature, max tokens, top_k sliders
- **Analytics Dashboard** (`ui/analytics_page.py`): Document metrics, confidence gauge, Plotly charts
- Sidebar navigation extended to include "Settings" and "Analytics"
- Chat page dynamically applies settings to QASystem on every render

---

## [Week 5] — 2026-02-26

### Added
- **Streamlit Application** (`app.py`): Main entrypoint with routing and session state
- **UI Components** in `ui/`:
  - `sidebar.py` — navigation menu with DB health indicator
  - `upload_page.py` — PDF upload with progress bars
  - `chat_page.py` — ChatGPT-style chat with source citations
  - `doc_viewer_page.py` — document search and deletion
- Installed `PyMuPDF` and `plotly`

---

## [Week 4] — 2026-02-26

### Added
- `modules/query_processor.py`: Semantic search with in-memory caching
- `modules/context_builder.py`: Token-aware chunk assembly with deduplication
- `modules/qa_system.py`: End-to-end Q&A pipeline with confidence scores
- `tests/test_week3_4.py`: Full 64-test suite (100% pass rate)

### Fixed
- `QASystem.answer_question()` incorrectly called non-existent `get_chat_completion()` → corrected to `generate_answer()`

---

## [Week 3] — 2026-02-15

### Added
- `modules/vector_db_setup.py`: Abstract VectorDB base class
- `modules/chromadb_handler.py`: Persistent ChromaDB implementation
- `modules/faiss_handler.py`: In-memory FAISS implementation with disk serialization
- `modules/storage_pipeline.py`: PDF → embed → store orchestration

---

## [Week 2] — 2026-02-15

### Added
- `modules/embedding_pipeline.py`: Batch embedding with SQLite disk cache
- `tests/test_week1_2.py`: Integration tests with real NIST PDFs
- Performance benchmarks for embedding batch sizes

---

## [Week 1] — 2026-02-15

### Added
- **Initial project structure**
- `modules/pdf_processor.py`: PyMuPDF-based text extraction with metadata
- `modules/text_cleaner.py`: OCR error correction, whitespace normalization
- `modules/text_chunker.py`: Sentence-aware chunking with overlap
- `modules/openai_handler.py`: FastRouter API wrapper with retry logic
- `requirements.txt`, `.env.example`, `README.md`, `.gitignore`
