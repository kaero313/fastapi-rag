# JSON 문자열/파일을 다루는 표준 라이브러리
import json
# 경로를 객체처럼 다루기 위한 표준 라이브러리
from pathlib import Path
# 백그라운드 작업을 위한 스레드
from threading import Thread

# FastAPI 핵심 클래스와 업로드 관련 타입
from fastapi import FastAPI, File, HTTPException, UploadFile
# 파일을 그대로 응답으로 반환하는 클래스
from fastapi.responses import FileResponse
# 정적 파일(CSS/JS)을 제공하는 유틸
from fastapi.staticfiles import StaticFiles

# 환경 설정값(.env)을 읽어오는 객체
from app.core.config import settings
# 디렉터리 인입 작업(비동기)의 상태 관리 함수들
from app.rag.ingest_jobs import (
    create_job,  # 새 작업 생성
    get_job,  # 작업 조회
    mark_completed,  # 작업 완료 처리
    mark_failed,  # 작업 실패 처리
    mark_running,  # 작업 실행 중 처리
    serialize_job,  # 작업 상태를 dict로 변환
)
# RAG 핵심 서비스 함수들
from app.rag.service import (
    answer_query,  # 질의 처리
    count_tokens as count_tokens_service,  # 토큰 카운트(이름 충돌 방지)
    get_sources,  # source 목록 조회
    ingest_directory,  # 디렉터리 인입
    ingest_documents,  # 문서 리스트 직접 인입
    ingest_json_bytes,  # JSON 업로드 인입
    ingest_pdf_bytes,  # PDF 업로드 인입
    reset_db,  # DB 초기화
)
# 요청/응답 스키마(데이터 구조) 정의
from app.schemas import (
    CountTokensRequest,  # /count-tokens 요청
    CountTokensResponse,  # /count-tokens 응답
    IngestDirectoryRequest,  # /ingest-dir 요청
    IngestRequest,  # /ingest 요청
    QueryRequest,  # /query 요청
    QueryResponse,  # /query 응답
    ResetDbRequest,  # /reset-db 요청
    ResetDbResponse,  # /reset-db 응답
    SourcesResponse,  # /sources 응답
)

# FastAPI 앱 생성
app = FastAPI(title="fastapi-rag")
# 현재 파일의 위치를 기준으로 경로 계산
BASE_DIR = Path(__file__).resolve().parent
# 정적 파일이 있는 폴더 경로
STATIC_DIR = BASE_DIR / "static"
# HTML 파일이 있는 폴더 경로
WEB_DIR = BASE_DIR / "web"

# /static 경로로 정적 파일 제공
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 메인 UI 페이지
@app.get("/")
def index():
    # index.html을 그대로 반환
    return FileResponse(WEB_DIR / "index.html")


# 서버 상태 확인용 헬스체크
@app.get("/health")
def health():
    # 간단한 상태 문자열 반환
    return {"status": "ok"}


# 문서 리스트를 직접 인입하는 엔드포인트
@app.post("/ingest")
def ingest(request: IngestRequest):
    # 문서들을 벡터 DB에 저장
    ids = ingest_documents(request.documents)
    # 인입 결과 반환
    return {"ingested": len(ids), "ids": ids}


# PDF 업로드 인입 엔드포인트
@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    # 파일 이름이 없거나 확장자가 PDF가 아니면 오류
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    # 업로드된 파일 바이트를 읽음
    content = await file.read()
    # 파일이 비어있으면 오류
    if not content:
        raise HTTPException(status_code=400, detail="Empty PDF file.")
    # PDF 텍스트를 추출해 인입 수행
    ids = ingest_pdf_bytes(file.filename, content)
    # 텍스트가 전혀 추출되지 않으면 오류
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF.",
        )
    # 인입 결과 반환
    return {"ingested": len(ids), "ids": ids}


# JSON 업로드 인입 엔드포인트
@app.post("/ingest-json")
async def ingest_json(file: UploadFile = File(...)):
    # 파일 이름이 없거나 확장자가 JSON이 아니면 오류
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported.")
    # 업로드된 파일 바이트를 읽음
    content = await file.read()
    # 파일이 비어있으면 오류
    if not content:
        raise HTTPException(status_code=400, detail="Empty JSON file.")
    try:
        # JSON을 문서로 변환 후 인입
        ids = ingest_json_bytes(file.filename, content)
    except json.JSONDecodeError as exc:
        # JSON 파싱이 실패하면 오류
        raise HTTPException(status_code=400, detail="Invalid JSON file.") from exc
    # 인입할 레코드가 없으면 오류
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="No ingestable records found in JSON.",
        )
    # 인입 결과 반환
    return {"ingested": len(ids), "ids": ids}


# 디렉터리 인입 엔드포인트(동기)
@app.post("/ingest-dir")
def ingest_dir(request: IngestDirectoryRequest):
    # 경로 검증 및 실제 대상 경로 계산
    target, base_dir = _resolve_ingest_target(request)
    # 디렉터리 인입 수행
    ids, stats = ingest_directory(
        target,
        recursive=request.recursive,
        extensions=request.extensions,
        base_dir=base_dir,
    )
    # 인입 결과가 없으면 오류
    if not ids:
        raise HTTPException(status_code=400, detail="No ingestable files found.")

    # 기본 응답 구성
    response = {"ingested": len(ids), "ids": ids}
    # 통계 정보 추가
    response.update(stats)
    return response


# 디렉터리 인입 엔드포인트(비동기)
@app.post("/ingest-dir-async")
def ingest_dir_async(request: IngestDirectoryRequest):
    # 경로 검증 및 실제 대상 경로 계산
    target, base_dir = _resolve_ingest_target(request)
    # 작업 객체 생성
    job = create_job()
    # 백그라운드 스레드로 인입 시작
    Thread(
        target=_run_ingest_dir_job,
        args=(job.id, target, request.recursive, request.extensions, base_dir),
        daemon=True,
    ).start()
    # 즉시 작업 ID와 상태 반환
    return {"job_id": job.id, "status": job.status.value}


# 디렉터리 인입 작업 상태 조회
@app.get("/ingest-dir-jobs/{job_id}")
def ingest_dir_job(job_id: str):
    # 작업 조회
    job = get_job(job_id)
    # 작업이 없으면 404
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    # 작업 상태를 dict로 반환
    return serialize_job(job)


# 질의 처리 엔드포인트
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    # RAG 질의 처리 함수 호출
    return answer_query(
        request.query,
        top_k=request.top_k,
        source=request.source,
        sources=request.sources,
        page_gte=request.page_gte,
        page_lte=request.page_lte,
    )


# source 목록 조회 엔드포인트
@app.get("/sources", response_model=SourcesResponse)
def sources(limit: int | None = None):
    # source 리스트를 응답 스키마에 담아 반환
    return SourcesResponse(sources=get_sources(limit=limit))


# 토큰 카운트 엔드포인트
@app.post("/count-tokens", response_model=CountTokensResponse)
def count_tokens(request: CountTokensRequest):
    # 모델 이름 결정(요청값이 없으면 기본값)
    model_name = request.model or settings.gemini_model
    # 텍스트가 비어있으면 0으로 처리
    if not request.text.strip():
        return CountTokensResponse(model=model_name, tokens=0)
    try:
        # Gemini API로 토큰 수 계산
        tokens = count_tokens_service(request.text, model=request.model)
    except Exception as exc:
        # 실패하면 502 에러로 반환
        raise HTTPException(
            status_code=502,
            detail=f"Token count failed: {exc}",
        ) from exc
    # 계산된 토큰 수 반환
    return CountTokensResponse(model=model_name, tokens=tokens)


# DB 초기화 엔드포인트
@app.post("/reset-db", response_model=ResetDbResponse)
def reset_database(request: ResetDbRequest):
    # confirm 플래그가 없으면 실수 방지 에러
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Reset not confirmed. Set confirm=true to proceed.",
        )
    # DB 초기화 수행
    result = reset_db()
    # 결과를 응답 스키마로 반환
    return ResetDbResponse(**result)


# 디렉터리 인입 대상 경로 계산 및 검증
def _resolve_ingest_target(
    request: IngestDirectoryRequest,
) -> tuple[Path, Path | None]:
    # ingest_base_dir 설정이 있으면 절대 경로로 변환
    base_dir = (
        Path(settings.ingest_base_dir).resolve()
        if settings.ingest_base_dir
        else None
    )
    # 요청으로 들어온 디렉터리 문자열을 Path로 변환
    directory = Path(request.directory)
    # 절대 경로면 그대로 사용
    if directory.is_absolute():
        target = directory.resolve()
    else:
        # 상대 경로면 base_dir 기준으로 합침
        target = (base_dir / directory).resolve() if base_dir else directory.resolve()

    # base_dir 밖의 경로면 차단
    if base_dir and not _is_within_base(target, base_dir):
        raise HTTPException(
            status_code=400,
            detail="Directory must be under INGEST_BASE_DIR.",
        )
    # 경로가 없거나 폴더가 아니면 오류
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found.")
    return target, base_dir


# 비동기 인입 작업의 실제 실행 함수
def _run_ingest_dir_job(
    job_id: str,
    target: Path,
    recursive: bool,
    extensions: list[str] | None,
    base_dir: Path | None,
) -> None:
    # 작업 상태를 실행 중으로 변경
    mark_running(job_id)
    try:
        # 디렉터리 인입 수행
        ids, stats = ingest_directory(
            target,
            recursive=recursive,
            extensions=extensions,
            base_dir=base_dir,
        )
        # 인입 결과가 없으면 실패 처리
        if not ids:
            raise ValueError("No ingestable files found.")
        # 결과 구성
        result = {"ingested": len(ids), "ids": ids}
        result.update(stats)
        # 완료 상태 기록
        mark_completed(job_id, result)
    except Exception as exc:
        # 실패 상태 기록
        mark_failed(job_id, str(exc))


# path가 base_dir 안에 있는지 확인
def _is_within_base(path: Path, base_dir: Path) -> bool:
    try:
        # base_dir 기준 상대 경로 계산 시도
        path.relative_to(base_dir)
        return True
    except ValueError:
        # base_dir 밖이면 예외가 발생
        return False
