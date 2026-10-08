# PDF AI Agent (RAG)

Upload a PDF, ask questions, get answers with page citations, spelling correction and related questions.

## Run it

```bash
cd pdf-ai-agent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then put your API key inside
uvicorn backend.main:app --reload
```

Open http://localhost:8000

The first upload downloads two small local models (embeddings and reranker), so it takes a minute once.

**Scanned PDFs (OCR):** also install the Tesseract program (Windows: UB-Mannheim installer, Mac: `brew install tesseract`, Linux: `apt install tesseract-ocr`). Without it, text PDFs still work.

## How it works

Upload -> PyMuPDF text (OCR if a page has no text) -> chunks of ~220 words with page number -> embeddings -> Qdrant (stored in `data/qdrant`, no Docker needed)

Question -> spelling fix (small LLM call) -> vector search + BM25 keyword search -> merge (RRF) -> rerank -> top 5 chunks -> LLM answers using only those chunks -> answer + pages + 3 related questions

| File | Job |
|---|---|
| `backend/ingest.py` | PDF reading, OCR, chunking, indexing, progress |
| `backend/search.py` | Hybrid search + reranking |
| `backend/llm.py` | Spelling correction, answer generation |
| `backend/main.py` | API: `/upload`, `/status`, `/ask`, `/pdf` |
| `static/index.html` | Web page |

## Before you claim "100,000 pages is fast"

This is an MVP. Test with 100, 1,000, then 10,000 pages and note upload/indexing time, search time and answer time (the UI shows these per question). To scale up:

1. Replace `BackgroundTasks` with Celery + Redis workers.
2. Run Qdrant as a server (Docker) instead of local-file mode, and store BM25 in Qdrant sparse vectors or Elasticsearch instead of in memory.
3. Parallelize OCR across CPU cores, and use a GPU or embedding API for faster indexing.
4. Add table-aware extraction and cross-page chunk overlap.
