import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.rag.service import (
    answer_query,
    ingest_directory,
    ingest_documents,
    ingest_json_bytes,
    ingest_pdf_bytes,
)
from app.schemas import (
    IngestDirectoryRequest,
    IngestRequest,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(title="fastapi-rag")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
WEB_DIR = BASE_DIR / "web"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    ids = ingest_documents(request.documents)
    return {"ingested": len(ids), "ids": ids}


@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty PDF file.")
    ids = ingest_pdf_bytes(file.filename, content)
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF.",
        )
    return {"ingested": len(ids), "ids": ids}


@app.post("/ingest-json")
async def ingest_json(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty JSON file.")
    try:
        ids = ingest_json_bytes(file.filename, content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON file.") from exc
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="No ingestable records found in JSON.",
        )
    return {"ingested": len(ids), "ids": ids}


@app.post("/ingest-dir")
def ingest_dir(request: IngestDirectoryRequest):
    base_dir = Path(settings.ingest_base_dir).resolve() if settings.ingest_base_dir else None
    directory = Path(request.directory)
    if directory.is_absolute():
        target = directory.resolve()
    else:
        target = (base_dir / directory).resolve() if base_dir else directory.resolve()

    if base_dir and not _is_within_base(target, base_dir):
        raise HTTPException(
            status_code=400,
            detail="Directory must be under INGEST_BASE_DIR.",
        )
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found.")

    ids, stats = ingest_directory(
        target,
        recursive=request.recursive,
        extensions=request.extensions,
        base_dir=base_dir,
    )
    if not ids:
        raise HTTPException(status_code=400, detail="No ingestable files found.")

    response = {"ingested": len(ids), "ids": ids}
    response.update(stats)
    return response


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return answer_query(request.query, top_k=request.top_k)


def _is_within_base(path: Path, base_dir: Path) -> bool:
    try:
        path.relative_to(base_dir)
        return True
    except ValueError:
        return False
