import json
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.rag.ingest_jobs import (
    create_job,
    get_job,
    mark_completed,
    mark_failed,
    mark_running,
    serialize_job,
)
from app.rag.service import (
    answer_query,
    count_tokens as count_tokens_service,
    ingest_directory,
    ingest_documents,
    ingest_json_bytes,
    ingest_pdf_bytes,
)
from app.schemas import (
    CountTokensRequest,
    CountTokensResponse,
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
    target, base_dir = _resolve_ingest_target(request)
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


@app.post("/ingest-dir-async")
def ingest_dir_async(request: IngestDirectoryRequest):
    target, base_dir = _resolve_ingest_target(request)
    job = create_job()
    Thread(
        target=_run_ingest_dir_job,
        args=(job.id, target, request.recursive, request.extensions, base_dir),
        daemon=True,
    ).start()
    return {"job_id": job.id, "status": job.status.value}


@app.get("/ingest-dir-jobs/{job_id}")
def ingest_dir_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return serialize_job(job)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return answer_query(request.query, top_k=request.top_k)


@app.post("/count-tokens", response_model=CountTokensResponse)
def count_tokens(request: CountTokensRequest):
    model_name = request.model or settings.gemini_model
    tokens = count_tokens_service(request.text, model=request.model)
    return CountTokensResponse(model=model_name, tokens=tokens)


def _resolve_ingest_target(
    request: IngestDirectoryRequest,
) -> tuple[Path, Path | None]:
    base_dir = (
        Path(settings.ingest_base_dir).resolve()
        if settings.ingest_base_dir
        else None
    )
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
    return target, base_dir


def _run_ingest_dir_job(
    job_id: str,
    target: Path,
    recursive: bool,
    extensions: list[str] | None,
    base_dir: Path | None,
) -> None:
    mark_running(job_id)
    try:
        ids, stats = ingest_directory(
            target,
            recursive=recursive,
            extensions=extensions,
            base_dir=base_dir,
        )
        if not ids:
            raise ValueError("No ingestable files found.")
        result = {"ingested": len(ids), "ids": ids}
        result.update(stats)
        mark_completed(job_id, result)
    except Exception as exc:
        mark_failed(job_id, str(exc))


def _is_within_base(path: Path, base_dir: Path) -> bool:
    try:
        path.relative_to(base_dir)
        return True
    except ValueError:
        return False
