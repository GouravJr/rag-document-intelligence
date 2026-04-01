"""
Configuration for the Intelligent Document Processor.
All settings are loaded from environment variables with sensible defaults.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_stores"
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# ── OCR ──
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")  # path to tesseract binary
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng+deu")     # English + German

# ── Embeddings ──
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"  # fast, free, no API key
)

# ── LLM (Groq — free tier) ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ── RAG ──
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# ── FastAPI ──
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
