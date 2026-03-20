"""
ingest_pdfs.py — One-time PDF ingestion script
===============================================
Copies PDFs from the source folder and ingests them into ChromaDB.

Run:
    venv\\Scripts\\python.exe ingest_pdfs.py
"""

import os
import sys
import shutil
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("ingest")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
PDF_DEST    = DATA_DIR / "pdfs"
CHROMA_DIR  = DATA_DIR / "chroma"

PDF_DEST.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

pdf_files = list(PDF_DEST.glob("*.pdf"))
if not pdf_files:
    logger.error(f"No PDFs found in {PDF_DEST}")
    sys.exit(1)

logger.info(f"Found {len(pdf_files)} PDF(s) in {PDF_DEST}")
copied = [str(p) for p in pdf_files]

# ── Ingest into ChromaDB ───────────────────────────────────────────────────
logger.info("\nInitialising ChromaDB...")
from src.modules.chromadb_handler import ChromaDBHandler
db = ChromaDBHandler(
    persist_directory=str(CHROMA_DIR),
    collection_name="document_chunks"
)
logger.info(f"Existing vectors: {db.get_count()}")
logger.info(f"Existing docs:    {db.list_documents()}")

logger.info("\nStarting ingestion pipeline...")
from src.modules.storage_pipeline import StoragePipeline
pipeline = StoragePipeline(
    vector_db=db,
    chunk_size=int(os.getenv("MAX_CHUNK_SIZE", 1000)),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
    skip_existing=True
)

stats = pipeline.process_pdfs(copied)

logger.info("\n=== FINAL DATABASE STATE ===")
logger.info(f"Total vectors : {db.get_count()}")
logger.info(f"Documents     : {db.list_documents()}")
