"""PDF -> text (OCR if needed) -> chunks -> embeddings -> Qdrant. Runs once per upload."""
import json
import uuid

import fitz  # PyMuPDF
from qdrant_client.models import PointStruct

from . import config as C
from .models import client, embedder

STATUS: dict[str, dict] = {}


def _write_status(doc_id: str):
    (C.DOCS / f"{doc_id}.status.json").write_text(json.dumps(STATUS[doc_id]))


def init_status(doc_id: str, filename: str):
    STATUS[doc_id] = {
        "doc_id": doc_id, "filename": filename, "stage": "queued", "progress": 0,
        "pages_done": 0, "total_pages": 0, "ocr_pages": 0, "chunks": 0, "error": None,
    }
    _write_status(doc_id)


def get_status(doc_id: str):
    if doc_id in STATUS:
        return STATUS[doc_id]
    f = C.DOCS / f"{doc_id}.status.json"
    return json.loads(f.read_text()) if f.exists() else None


def _update(doc_id: str, **kw):
    STATUS[doc_id].update(kw)
    _write_status(doc_id)


def ocr_page(page) -> str:
    """OCR a scanned page. Returns '' if Tesseract isn't installed."""
    try:
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def chunk_page(text: str, page: int) -> list[dict]:
    words = text.split()
    step = C.CHUNK_WORDS - C.CHUNK_OVERLAP
    out = []
    for n, i in enumerate(range(0, len(words), step)):
        piece = words[i:i + C.CHUNK_WORDS]
        out.append({"chunk_id": f"{page}_{n}", "page": page, "text": " ".join(piece)})
        if i + C.CHUNK_WORDS >= len(words):
            break
    return out


def process_pdf(doc_id: str, path: str):
    try:
        pdf = fitz.open(path)
        total = len(pdf)
        _update(doc_id, stage="reading pages", total_pages=total)

        chunks, ocr_pages = [], 0
        for i, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if len(text) < 30:  # likely a scanned page
                text = ocr_page(page).strip()
                if text:
                    ocr_pages += 1
            chunks += chunk_page(text, i)
            if i % 25 == 0 or i == total:
                _update(doc_id, pages_done=i, ocr_pages=ocr_pages,
                        progress=int(50 * i / total))

        if not chunks:
            raise ValueError("No text found. If this is a scanned PDF, install Tesseract for OCR.")

        _update(doc_id, stage="building search index", chunks=len(chunks))
        emb, qc, B = embedder(), client(), 16
        for s in range(0, len(chunks), B):
            batch = chunks[s:s + B]
            vecs = list(emb.embed([c["text"] for c in batch]))
            qc.upsert(C.COLLECTION, points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{c['chunk_id']}")),
                    vector=v.tolist(),
                    payload={"doc_id": doc_id, **c},
                ) for c, v in zip(batch, vecs)
            ])
            _update(doc_id, progress=50 + int(50 * min(s + B, len(chunks)) / len(chunks)))

        # Chunk text is kept on disk for BM25 keyword search.
        (C.DOCS / f"{doc_id}.chunks.json").write_text(json.dumps(chunks))
        _update(doc_id, stage="ready", progress=100)
    except Exception as e:
        _update(doc_id, stage="error", error=str(e))
