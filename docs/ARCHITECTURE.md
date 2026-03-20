# Architecture Overview
## AI File Assistant

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                          │
│  ┌──────────────┐ ┌──────────────────────────────────────────────┐ │
│  │ Sidebar Nav  │ │  Pages                                        │ │
│  │  sidebar.py  │ │ • chat_page.py    (Ask Questions)             │ │
│  └──────────────┘ │ • upload_page.py  (File Upload)               │ │
│                   │ • doc_viewer_page.py (Document Management)    │ │
│                   │ • analytics_page.py  (Analytics Dashboard)    │ │
│                   │ • settings_page.py   (Settings)               │ │
│                   └──────────────────────────────────────────────┘ │
└────────────────────────────┬───────────────────────────────────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │          Backend Pipeline Layer          │
        │                                          │
        │  ┌──────────────────────────────────┐   │
        │  │  storage_pipeline.py             │   │
        │  │  PDF → Clean → Chunk → Embed → DB│   │
        │  └─────────────────┬────────────────┘   │
        │                    │ (for queries)        │
        │  ┌─────────────────▼────────────────┐   │
        │  │  query_processor.py              │   │
        │  │  query_embedding → similarity    │   │
        │  └─────────────────┬────────────────┘   │
        │                    │                     │
        │  ┌─────────────────▼────────────────┐   │
        │  │  context_builder.py              │   │
        │  │  Chunk assembly, dedup, limits   │   │
        │  └─────────────────┬────────────────┘   │
        │                    │                     │
        │  ┌─────────────────▼────────────────┐   │
        │  │  qa_system.py                    │   │
        │  │  Orchestrates full Q&A pipeline  │   │
        │  └──────────────────────────────────┘   │
        └────────────────────────────────────────-┘
                             │
        ┌────────────────────▼────────────────────┐
        │          Core Module Layer               │
        │                                          │
        │  pdf_processor.py   → PDF text extraction│
        │  text_cleaner.py    → Text normalization │
        │  text_chunker.py    → Sentence-aware     │
        │                       chunking           │
        │  embedding_pipeline.py → Batch embedding │
        │                         with disk cache  │
        │  openai_handler.py  → API abstraction    │
        └─────────────────────────────────────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │          Vector Database Layer           │
        │                                          │
        │  vector_db_setup.py → Abstract VectorDB  │
        │  ┌─────────────────┐ ┌─────────────────┐ │
        │  │chromadb_handler │ │ faiss_handler   │ │
        │  │ (Persistent)    │ │ (In-Memory/Disk)│ │
        │  └─────────────────┘ └─────────────────┘ │
        └─────────────────────────────────────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │           External APIs                  │
        │                                          │
        │  FastRouter API (OpenAI-compatible proxy)│
        │  • /embeddings  → text-embedding-ada-002 │
        │  • /chat/completions → GPT / Claude      │
        └─────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
app.py
  ├── ui/sidebar.py
  ├── ui/chat_page.py       ← qa_system.py
  ├── ui/upload_page.py     ← storage_pipeline.py
  ├── ui/doc_viewer_page.py ← chromadb_handler.py
  ├── ui/analytics_page.py
  └── ui/settings_page.py

storage_pipeline.py
  ├── pdf_processor.py
  ├── text_cleaner.py
  ├── text_chunker.py
  ├── embedding_pipeline.py ← openai_handler.py
  └── vector_db_setup.py (abstract)

qa_system.py
  ├── query_processor.py   ← openai_handler.py, vector_db_setup.py
  ├── context_builder.py
  └── openai_handler.py
```

---

## Data Flow

### Ingestion Flow
```
PDF File → PDFProcessor (PyMuPDF) → TextCleaner → TextChunker
  → EmbeddingPipeline (API call) → ChromaDBHandler (upsert with metadata)
```

### Query Flow
```
User Question → QueryProcessor (embedding + cosine similarity)
  → Top-K chunks → ContextBuilder (token-aware assembly)
  → QASystem (GPT prompt) → Answer + Sources + Confidence
```

---

## Technology Stack

| Category | Technology |
|---------|-----------|
| UI | Streamlit 1.x |
| Vector DB | ChromaDB (persistent), FAISS (alternative) |
| PDF Processing | PyMuPDF (fitz) |
| Embeddings | OpenAI text-embedding-ada-002 via FastRouter |
| LLM | GPT-3.5-turbo / GPT-4 via FastRouter |
| Caching | SQLite (embedding cache), st.session_state |
| Testing | pytest |
| Deployment | Docker, Streamlit Cloud, GitHub Actions |
