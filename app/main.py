# JSON 문자열/파일을 다루는 표준 라이브러리입니다.
import json
# 경로를 객체처럼 다루는 표준 라이브러리입니다.
from pathlib import Path
# 백그라운드 작업에 사용할 스레드 클래스입니다.
from threading import Thread

# FastAPI 핵심 클래스와 업로드 관련 타입을 가져옵니다.
from fastapi import FastAPI, File, HTTPException, UploadFile
# 파일을 응답으로 반환하는 클래스를 가져옵니다.
from fastapi.responses import FileResponse
# 정적 파일(CSS/JS) 제공 유틸을 가져옵니다.
from fastapi.staticfiles import StaticFiles

# 환경설정 값을 가져옵니다.
from app.core.config import settings
# 비동기 인입 작업 상태 관리 함수를 가져옵니다.
from app.rag.ingest_jobs import (
    # 새 작업을 생성합니다.
    create_job,
    # 작업 상태를 조회합니다.
    get_job,
    # 작업 완료로 표시합니다.
    mark_completed,
    # 작업 실패로 표시합니다.
    mark_failed,
    # 작업 실행 중으로 표시합니다.
    mark_running,
    # 작업 상태를 dict로 직렬화합니다.
    serialize_job,
# 여러 줄 구문을 닫습니다.
)
# RAG 핵심 서비스 함수를 가져옵니다.
from app.rag.service import (
    # 질의를 처리해 답변을 생성합니다.
    answer_query,
    # 토큰 수 계산 함수를 가져옵니다.
    count_tokens as count_tokens_service,
    # 저장된 source 목록을 가져옵니다.
    get_sources,
    # 디렉터리 인입 함수를 가져옵니다.
    ingest_directory,
    # 문서 리스트 인입 함수를 가져옵니다.
    ingest_documents,
    # JSON 업로드 인입 함수를 가져옵니다.
    ingest_json_bytes,
    # PDF 업로드 인입 함수를 가져옵니다.
    ingest_pdf_bytes,
    # DB 초기화 함수를 가져옵니다.
    reset_db,
# 여러 줄 구문을 닫습니다.
)
# 요청/응답 스키마를 가져옵니다.
from app.schemas import (
    # /count-tokens 요청 스키마입니다.
    CountTokensRequest,
    # /count-tokens 응답 스키마입니다.
    CountTokensResponse,
    # /ingest-dir 요청 스키마입니다.
    IngestDirectoryRequest,
    # /ingest 요청 스키마입니다.
    IngestRequest,
    # /query 요청 스키마입니다.
    QueryRequest,
    # /query 응답 스키마입니다.
    QueryResponse,
    # /reset-db 요청 스키마입니다.
    ResetDbRequest,
    # /reset-db 응답 스키마입니다.
    ResetDbResponse,
    # /sources 응답 스키마입니다.
    SourcesResponse,
# 여러 줄 구문을 닫습니다.
)

# FastAPI 앱을 생성합니다.
app = FastAPI(title="fastapi-rag")
# 현재 파일 기준의 폴더 경로입니다.
BASE_DIR = Path(__file__).resolve().parent
# 정적 파일 폴더 경로입니다.
STATIC_DIR = BASE_DIR / "static"
# HTML 파일 폴더 경로입니다.
WEB_DIR = BASE_DIR / "web"

# /static 경로로 정적 파일 제공을 등록합니다.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 메인 UI 라우트입니다.
@app.get("/")
def index():  # index 함수를 정의합니다.
    # 값을 반환합니다.
    return FileResponse(WEB_DIR / "index.html")


# 헬스체크 라우트입니다.
@app.get("/health")
def health():  # health 함수를 정의합니다.
    # 값을 반환합니다.
    return {"status": "ok"}


# 문서 인입 라우트입니다.
@app.post("/ingest")
def ingest(request: IngestRequest):  # ingest 함수를 정의합니다.
    # ids에 값을 대입합니다.
    ids = ingest_documents(request.documents)
    # 값을 반환합니다.
    return {"ingested": len(ids), "ids": ids}


# PDF 업로드 인입 라우트입니다.
@app.post("/ingest-pdf")
# async def ingest_pdf(file: UploadFile에 값을 대입합니다.
async def ingest_pdf(file: UploadFile = File(...)):
    # 조건을 검사합니다.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        # raise HTTPException(status_code에 값을 대입합니다.
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    # content에 값을 대입합니다.
    content = await file.read()
    # 조건을 검사합니다.
    if not content:
        # raise HTTPException(status_code에 값을 대입합니다.
        raise HTTPException(status_code=400, detail="Empty PDF file.")
    # ids에 값을 대입합니다.
    ids = ingest_pdf_bytes(file.filename, content)
    # 조건을 검사합니다.
    if not ids:
        # 여러 줄 호출을 시작합니다.
        raise HTTPException(
            # status_code 인자를 전달합니다.
            status_code=400,
            # detail 인자를 전달합니다.
            detail="No extractable text found in PDF.",
        # 여러 줄 구문을 닫습니다.
        )
    # 값을 반환합니다.
    return {"ingested": len(ids), "ids": ids}


# JSON 업로드 인입 라우트입니다.
@app.post("/ingest-json")
# async def ingest_json(file: UploadFile에 값을 대입합니다.
async def ingest_json(file: UploadFile = File(...)):
    # 조건을 검사합니다.
    if not file.filename or not file.filename.lower().endswith(".json"):
        # raise HTTPException(status_code에 값을 대입합니다.
        raise HTTPException(status_code=400, detail="Only JSON files are supported.")
    # content에 값을 대입합니다.
    content = await file.read()
    # 조건을 검사합니다.
    if not content:
        # raise HTTPException(status_code에 값을 대입합니다.
        raise HTTPException(status_code=400, detail="Empty JSON file.")
    # 코드를 실행합니다.
    try:
        # ids에 값을 대입합니다.
        ids = ingest_json_bytes(file.filename, content)
    # 코드를 실행합니다.
    except json.JSONDecodeError as exc:
        # raise HTTPException(status_code에 값을 대입합니다.
        raise HTTPException(status_code=400, detail="Invalid JSON file.") from exc
    # 조건을 검사합니다.
    if not ids:
        # 여러 줄 호출을 시작합니다.
        raise HTTPException(
            # status_code 인자를 전달합니다.
            status_code=400,
            # detail 인자를 전달합니다.
            detail="No ingestable records found in JSON.",
        # 여러 줄 구문을 닫습니다.
        )
    # 값을 반환합니다.
    return {"ingested": len(ids), "ids": ids}


# 디렉터리 인입 라우트입니다.
@app.post("/ingest-dir")
def ingest_dir(request: IngestDirectoryRequest):  # ingest_dir 함수를 정의합니다.
    # target, base_dir에 값을 대입합니다.
    target, base_dir = _resolve_ingest_target(request)
    # ids, stats에 대입하는 호출을 시작합니다.
    ids, stats = ingest_directory(
        # 다음 인자를 전달합니다.
        target,
        # recursive 인자를 전달합니다.
        recursive=request.recursive,
        # extensions 인자를 전달합니다.
        extensions=request.extensions,
        # base_dir 인자를 전달합니다.
        base_dir=base_dir,
    # 여러 줄 구문을 닫습니다.
    )
    # 조건을 검사합니다.
    if not ids:
        # raise HTTPException(status_code에 값을 대입합니다.
        raise HTTPException(status_code=400, detail="No ingestable files found.")

    # response에 값을 대입합니다.
    response = {"ingested": len(ids), "ids": ids}
    # 코드를 실행합니다.
    response.update(stats)
    # 값을 반환합니다.
    return response


# 디렉터리 비동기 인입 라우트입니다.
@app.post("/ingest-dir-async")
def ingest_dir_async(request: IngestDirectoryRequest):  # ingest_dir_async 함수를 정의합니다.
    # target, base_dir에 값을 대입합니다.
    target, base_dir = _resolve_ingest_target(request)
    # job에 값을 대입합니다.
    job = create_job()
    # 여러 줄 호출을 시작합니다.
    Thread(
        # target 인자를 전달합니다.
        target=_run_ingest_dir_job,
        # args 인자를 전달합니다.
        args=(job.id, target, request.recursive, request.extensions, base_dir),
        # daemon 인자를 전달합니다.
        daemon=True,
    # 코드를 실행합니다.
    ).start()
    # 값을 반환합니다.
    return {"job_id": job.id, "status": job.status.value}


# 인입 작업 상태 조회 라우트입니다.
@app.get("/ingest-dir-jobs/{job_id}")
def ingest_dir_job(job_id: str):  # ingest_dir_job 함수를 정의합니다.
    # job에 값을 대입합니다.
    job = get_job(job_id)
    # 조건을 검사합니다.
    if job is None:
        # raise HTTPException(status_code에 값을 대입합니다.
        raise HTTPException(status_code=404, detail="Job not found.")
    # 값을 반환합니다.
    return serialize_job(job)


# 질의 처리 라우트입니다.
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):  # query 함수를 정의합니다.
    # 여러 줄 호출을 시작합니다.
    return answer_query(
        # 다음 인자를 전달합니다.
        request.query,
        # top_k 인자를 전달합니다.
        top_k=request.top_k,
        # source 인자를 전달합니다.
        source=request.source,
        # sources 인자를 전달합니다.
        sources=request.sources,
        # page_gte 인자를 전달합니다.
        page_gte=request.page_gte,
        # page_lte 인자를 전달합니다.
        page_lte=request.page_lte,
    # 여러 줄 구문을 닫습니다.
    )


# source 목록 조회 라우트입니다.
@app.get("/sources", response_model=SourcesResponse)
def sources(limit: int | None = None):  # sources 함수를 정의합니다.
    # 값을 반환합니다.
    return SourcesResponse(sources=get_sources(limit=limit))


# 토큰 카운트 라우트입니다.
@app.post("/count-tokens", response_model=CountTokensResponse)
def count_tokens(request: CountTokensRequest):  # count_tokens 함수를 정의합니다.
    # model_name에 값을 대입합니다.
    model_name = request.model or settings.gemini_model
    # 조건을 검사합니다.
    if not request.text.strip():
        # 값을 반환합니다.
        return CountTokensResponse(model=model_name, tokens=0)
    # 코드를 실행합니다.
    try:
        # tokens에 값을 대입합니다.
        tokens = count_tokens_service(request.text, model=request.model)
    # 코드를 실행합니다.
    except Exception as exc:
        # 여러 줄 호출을 시작합니다.
        raise HTTPException(
            # status_code 인자를 전달합니다.
            status_code=502,
            # detail 인자를 전달합니다.
            detail=f"Token count failed: {exc}",
        # 코드를 실행합니다.
        ) from exc
    # 값을 반환합니다.
    return CountTokensResponse(model=model_name, tokens=tokens)


# DB 초기화 라우트입니다.
@app.post("/reset-db", response_model=ResetDbResponse)
def reset_database(request: ResetDbRequest):  # reset_database 함수를 정의합니다.
    # 조건을 검사합니다.
    if not request.confirm:
        # 여러 줄 호출을 시작합니다.
        raise HTTPException(
            # status_code 인자를 전달합니다.
            status_code=400,
            # detail 인자를 전달합니다.
            detail="Reset not confirmed. Set confirm=true to proceed.",
        # 여러 줄 구문을 닫습니다.
        )
    # result에 값을 대입합니다.
    result = reset_db()
    # 값을 반환합니다.
    return ResetDbResponse(**result)


# _resolve_ingest_target 함수를 정의합니다.
def _resolve_ingest_target(
    # request 매개변수입니다.
    request: IngestDirectoryRequest,
# 함수 시그니처를 닫고 반환 타입을 적습니다.
) -> tuple[Path, Path | None]:
    # base_dir에 대입하는 호출을 시작합니다.
    base_dir = (
        # 코드를 실행합니다.
        Path(settings.ingest_base_dir).resolve()
        # 조건을 검사합니다.
        if settings.ingest_base_dir
        # 코드를 실행합니다.
        else None
    # 여러 줄 구문을 닫습니다.
    )
    # directory에 값을 대입합니다.
    directory = Path(request.directory)
    # 조건을 검사합니다.
    if directory.is_absolute():
        # target에 값을 대입합니다.
        target = directory.resolve()
    # 이전 조건이 거짓일 때 실행합니다.
    else:
        # target에 값을 대입합니다.
        target = (base_dir / directory).resolve() if base_dir else directory.resolve()

    # 조건을 검사합니다.
    if base_dir and not _is_within_base(target, base_dir):
        # 여러 줄 호출을 시작합니다.
        raise HTTPException(
            # status_code 인자를 전달합니다.
            status_code=400,
            # detail 인자를 전달합니다.
            detail="Directory must be under INGEST_BASE_DIR.",
        # 여러 줄 구문을 닫습니다.
        )
    # 조건을 검사합니다.
    if not target.exists() or not target.is_dir():
        # raise HTTPException(status_code에 값을 대입합니다.
        raise HTTPException(status_code=404, detail="Directory not found.")
    # 값을 반환합니다.
    return target, base_dir


# _run_ingest_dir_job 함수를 정의합니다.
def _run_ingest_dir_job(
    # job_id 매개변수입니다.
    job_id: str,
    # target 매개변수입니다.
    target: Path,
    # recursive 매개변수입니다.
    recursive: bool,
    # extensions 매개변수입니다.
    extensions: list[str] | None,
    # base_dir 매개변수입니다.
    base_dir: Path | None,
# 함수 시그니처를 닫고 반환 타입을 적습니다.
) -> None:
    # 코드를 실행합니다.
    mark_running(job_id)
    # 코드를 실행합니다.
    try:
        # ids, stats에 대입하는 호출을 시작합니다.
        ids, stats = ingest_directory(
            # 다음 인자를 전달합니다.
            target,
            # recursive 인자를 전달합니다.
            recursive=recursive,
            # extensions 인자를 전달합니다.
            extensions=extensions,
            # base_dir 인자를 전달합니다.
            base_dir=base_dir,
        # 여러 줄 구문을 닫습니다.
        )
        # 조건을 검사합니다.
        if not ids:
            # 코드를 실행합니다.
            raise ValueError("No ingestable files found.")
        # result에 값을 대입합니다.
        result = {"ingested": len(ids), "ids": ids}
        # 코드를 실행합니다.
        result.update(stats)
        # 코드를 실행합니다.
        mark_completed(job_id, result)
    # 코드를 실행합니다.
    except Exception as exc:
        # 코드를 실행합니다.
        mark_failed(job_id, str(exc))


# _is_within_base 함수를 정의합니다.
def _is_within_base(path: Path, base_dir: Path) -> bool:
    # 코드를 실행합니다.
    try:
        # 코드를 실행합니다.
        path.relative_to(base_dir)
        # 값을 반환합니다.
        return True
    # 코드를 실행합니다.
    except ValueError:
        # 값을 반환합니다.
        return False
