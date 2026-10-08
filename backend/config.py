import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
DOCS = DATA / "docs"
QDRANT_PATH = DATA / "qdrant"
for p in (UPLOADS, DOCS, QDRANT_PATH):
    p.mkdir(parents=True, exist_ok=True)

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "gemini-3.8-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Retrieval models (run locally, no API key needed)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# Chunking (measured in words; roughly 1 word = 1.3 tokens)
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40

COLLECTION = "pdf_chunks"
