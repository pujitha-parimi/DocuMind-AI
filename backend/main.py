import shutil
import time
import uuid

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as C
from . import ingest, llm, search

app = FastAPI(title="PDF AI Agent")
app.mount("/static", StaticFiles(directory=C.BASE / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(C.BASE / "static" / "index.html")


@app.post("/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    doc_id = uuid.uuid4().hex[:12]
    path = C.UPLOADS / f"{doc_id}.pdf"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    ingest.init_status(doc_id, file.filename)
    # Runs in the background so the upload returns immediately.
    # For production scale, replace with a Celery/RQ worker.
    background.add_task(ingest.process_pdf, doc_id, str(path))
    return {"doc_id": doc_id}


@app.get("/status/{doc_id}")
def status(doc_id: str):
    s = ingest.get_status(doc_id)
    if not s:
        raise HTTPException(404, "Unknown document.")
    return s


@app.get("/pdf/{doc_id}")
def pdf(doc_id: str):
    path = C.UPLOADS / f"{doc_id}.pdf"
    if not path.exists():
        raise HTTPException(404, "PDF not found.")
    return FileResponse(path, media_type="application/pdf")


class Ask(BaseModel):
    doc_id: str
    question: str
    correct: bool = True


@app.post("/ask")
def ask(body: Ask):
    s = ingest.get_status(body.doc_id)
    if not s or s["stage"] != "ready":
        raise HTTPException(409, "The document is not ready yet.")
    original = body.question.strip()
    if not original:
        raise HTTPException(400, "Type a question first.")

    t0 = time.perf_counter()
    corrected, changed = llm.correct_query(original) if body.correct else (original, False)
    t1 = time.perf_counter()
    chunks = search.retrieve(body.doc_id, corrected)
    t2 = time.perf_counter()
    result = llm.answer(corrected, chunks)
    t3 = time.perf_counter()

    seen, sources = set(), []
    for c in chunks:
        if c["page"] not in seen:
            seen.add(c["page"])
            sources.append({"page": c["page"], "snippet": c["text"][:220]})

    return {
        "original": original, "corrected": corrected, "changed": changed,
        "answer": result["answer"], "related_questions": result["related_questions"],
        "sources": sources,
        "timings": {"spellcheck": round(t1 - t0, 2), "search": round(t2 - t1, 2),
                    "answer": round(t3 - t2, 2), "total": round(t3 - t0, 2)},
    }
