# Beginner Code Walkthrough (Line by Line)

이 문서는 **파이썬을 처음 배우는 사람**이 코드를 보면서 이해할 수 있도록,
각 파일을 **줄(라인) 단위로 설명**합니다.

> 대상: Python 서버 코드(백엔드). 프론트(HTML/CSS/JS) 설명은 필요하면 추가 가능.

---

## 1) `app/core/config.py`

1) `from pydantic_settings import BaseSettings, SettingsConfigDict`
- Pydantic 설정 클래스를 가져옵니다. 환경변수(.env)를 읽는 데 사용됩니다.

2) *(빈 줄)*
- 코드 가독성을 위한 줄바꿈입니다.

3) *(빈 줄)*
- 클래스 정의와 임포트 구분을 위한 여백입니다.

4) `class Settings(BaseSettings):`
- 환경변수를 읽는 설정 클래스를 선언합니다.

5) `gemini_api_key: str`
- 필수 설정: Gemini API Key 문자열입니다.

6) `gemini_model: str = "gemini-2.5-flash"`
- 기본 LLM 모델 이름입니다. .env에서 바꿀 수 있습니다.

7) `gemini_embedding_model: str = "text-embedding-004"`
- 임베딩(벡터) 모델 이름입니다.

8) `embed_max_retries: int = 3`
- 임베딩 호출 실패 시 재시도 횟수입니다.

9) `embed_retry_backoff: float = 1.0`
- 재시도 시 대기 시간(초)의 기본값입니다.

10) `ingest_batch_size: int = 64`
- 문서 인입 시 한 번에 처리할 문서 수입니다.

11) `ingest_chunk_size: int = 4000`
- 청킹 기준 토큰 수(근사치)입니다.

12) `ingest_chunk_overlap: int = 400`
- 청크 간 겹치는 토큰 수입니다.

13) `candidate_k_multiplier: int = 10`
- 검색 후보 개수를 `top_k * multiplier`로 늘리는 값입니다.

14) `candidate_k_min: int = 100`
- 후보 개수의 최소값입니다.

15) `chroma_persist_dir: str = "data/chroma"`
- Chroma DB가 저장되는 폴더입니다.

16) `chroma_collection: str = "rag"`
- Chroma 컬렉션 이름입니다.

17) `top_k: int = 4`
- 기본 검색 결과 개수입니다.

18) `ingest_base_dir: str = "data/ingest"`
- 디렉터리 인입 시 기준이 되는 폴더입니다.

19) *(빈 줄)*
- 아래 설정 구분용 줄바꿈입니다.

20) `model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")`
- `.env` 파일을 읽어 설정을 덮어쓰도록 합니다.

21) *(빈 줄)*
- 클래스 정의 종료 여백입니다.

22) *(빈 줄)*
- 아래 인스턴스 생성과 분리된 여백입니다.

23) `settings = Settings()`
- Settings 클래스를 실제로 생성합니다. 이제 `settings.xxx`로 설정 사용 가능합니다.

---

## 2) `app/schemas.py`

1) `from typing import Any`
- `Any`는 “어떤 타입이든 가능”이라는 의미입니다.

2) *(빈 줄)*
- 가독성을 위한 줄바꿈입니다.

3) `from pydantic import BaseModel`
- Pydantic의 기본 모델 클래스를 가져옵니다.

4) *(빈 줄)*
- 줄바꿈입니다.

5) *(빈 줄)*
- 클래스 구분용 여백입니다.

6) `class DocumentIn(BaseModel):`
- 문서 입력을 정의하는 모델입니다.

7) `id: str | None = None`
- 문서 ID(선택). 없으면 자동 생성됩니다.

8) `text: str`
- 문서 본문 텍스트입니다.

9) `metadata: dict[str, Any] | None = None`
- 메타데이터(출처, 페이지 등)를 담는 딕셔너리입니다.

10) *(빈 줄)*
- 줄바꿈입니다.

11) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

12) `class IngestRequest(BaseModel):`
- `/ingest` 요청 바디 모델입니다.

13) `documents: list[DocumentIn]`
- 여러 문서를 리스트로 받습니다.

14) *(빈 줄)*
- 줄바꿈입니다.

15) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

16) `class QueryRequest(BaseModel):`
- `/query` 요청 바디 모델입니다.

17) `query: str`
- 사용자 질문 문자열입니다.

18) `top_k: int | None = None`
- 선택적으로 검색 결과 수를 지정합니다.

19) `source: str | None = None`
- 단일 source 필터입니다.

20) `sources: list[str] | None = None`
- 복수 source 필터입니다.

21) `page_gte: int | None = None`
- 페이지 필터 최소값입니다.

22) `page_lte: int | None = None`
- 페이지 필터 최대값입니다.

23) *(빈 줄)*
- 줄바꿈입니다.

24) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

25) `class IngestDirectoryRequest(BaseModel):`
- `/ingest-dir` 요청 바디 모델입니다.

26) `directory: str`
- 인입할 디렉터리 경로(기본 폴더 기준)입니다.

27) `recursive: bool = True`
- 하위 폴더까지 탐색할지 여부입니다.

28) `extensions: list[str] | None = None`
- 특정 확장자만 인입하고 싶을 때 사용합니다.

29) *(빈 줄)*
- 줄바꿈입니다.

30) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

31) `class CountTokensRequest(BaseModel):`
- `/count-tokens` 요청 모델입니다.

32) `text: str`
- 토큰 수를 알고 싶은 텍스트입니다.

33) `model: str | None = None`
- 토큰 카운트에 사용할 모델 이름입니다.

34) *(빈 줄)*
- 줄바꿈입니다.

35) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

36) `class Source(BaseModel):`
- 검색 결과에 포함되는 “소스 문서” 구조입니다.

37) `id: str`
- 문서(청크) ID입니다.

38) `score: float`
- 유사도 점수입니다.

39) `text: str`
- 문서 내용 일부입니다.

40) `metadata: dict[str, Any] | None = None`
- 메타데이터입니다.

41) *(빈 줄)*
- 줄바꿈입니다.

42) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

43) `class QueryResponse(BaseModel):`
- `/query` 응답 모델입니다.

44) `answer: str`
- LLM이 생성한 답변입니다.

45) `sources: list[Source]`
- 근거로 사용된 소스 목록입니다.

46) *(빈 줄)*
- 줄바꿈입니다.

47) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

48) `class CountTokensResponse(BaseModel):`
- `/count-tokens` 응답 모델입니다.

49) `model: str`
- 토큰 카운트를 계산한 모델 이름입니다.

50) `tokens: int`
- 계산된 토큰 수입니다.

51) *(빈 줄)*
- 줄바꿈입니다.

52) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

53) `class SourcesResponse(BaseModel):`
- `/sources` 응답 모델입니다.

54) `sources: list[str]`
- 저장된 source 목록입니다.

55) *(빈 줄)*
- 줄바꿈입니다.

56) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

57) `class ResetDbRequest(BaseModel):`
- `/reset-db` 요청 모델입니다.

58) `confirm: bool = False`
- 실수 방지용 확인 플래그입니다.

59) *(빈 줄)*
- 줄바꿈입니다.

60) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

61) `class ResetDbResponse(BaseModel):`
- `/reset-db` 응답 모델입니다.

62) `collection: str`
- 초기화된 컬렉션 이름입니다.

63) `before_count: int`
- 초기화 전 문서 수입니다.

64) `after_count: int`
- 초기화 후 문서 수입니다.

65) `reset: bool`
- 초기화 성공 여부입니다.

66) `errors: list[str] | None = None`
- 초기화 과정 중 오류 메시지 목록입니다.

---

## 3) `app/main.py`

(이 파일은 라인이 많아서 “블록 + 줄 설명” 형태로 설명합니다.)

### [A] 임포트/앱 초기화

- `import json`
  - JSON 파싱용 표준 라이브러리입니다.
- `from pathlib import Path`
  - 파일 경로를 다루기 위한 표준 라이브러리입니다.
- `from threading import Thread`
  - 비동기 인입을 위한 스레드 실행에 사용합니다.
- `from fastapi import FastAPI, File, HTTPException, UploadFile`
  - FastAPI 기본 클래스와 파일 업로드 관련 타입입니다.
- `from fastapi.responses import FileResponse`
  - HTML 파일 반환에 사용합니다.
- `from fastapi.staticfiles import StaticFiles`
  - `/static` 경로에 정적 파일 제공을 위해 사용합니다.

- `from app.core.config import settings`
  - `.env` 기반 설정을 가져옵니다.

- `from app.rag.ingest_jobs import ...`
  - 디렉터리 비동기 인입 관련 작업 함수들입니다.

- `from app.rag.service import ...`
  - 인입, 질의, 토큰 카운트, DB 리셋 등 핵심 함수입니다.

- `from app.schemas import ...`
  - 요청/응답 데이터 모델을 가져옵니다.

- `app = FastAPI(title="fastapi-rag")`
  - FastAPI 애플리케이션을 생성합니다.

- `BASE_DIR = Path(__file__).resolve().parent`
  - `app/` 폴더 기준 경로를 구합니다.

- `STATIC_DIR = BASE_DIR / "static"`
  - 정적 파일(CSS/JS) 경로입니다.

- `WEB_DIR = BASE_DIR / "web"`
  - HTML 파일이 있는 경로입니다.

- `app.mount("/static", StaticFiles(...), name="static")`
  - `/static` URL로 CSS/JS를 제공하도록 등록합니다.

### [B] 기본 라우트

- `@app.get("/")` → `index()`
  - 메인 UI HTML 파일을 반환합니다.

- `@app.get("/health")`
  - 서버가 살아 있는지 확인하는 헬스체크입니다.

### [C] 인입 관련 라우트

- `/ingest` : JSON 문서 리스트를 직접 인입
- `/ingest-pdf` : PDF 업로드 → 텍스트 추출 → 인입
- `/ingest-json` : JSON 업로드 → 문서 변환 → 인입
- `/ingest-dir` : 폴더 전체 파일 인입
- `/ingest-dir-async` : 백그라운드 인입 시작
- `/ingest-dir-jobs/{job_id}` : 비동기 작업 상태 조회

각 함수는 공통적으로:
1) 입력 검증
2) 서비스 함수 호출
3) 결과 JSON 반환

### [D] 검색/기타

- `/sources` : 저장된 source 목록 반환
- `/query` : 질의 처리
- `/count-tokens` : 토큰 카운트
- `/reset-db` : DB 초기화

### [E] 내부 헬퍼 함수

- `_resolve_ingest_target()`
  - 디렉터리 경로가 안전한지 확인하고 실제 경로를 계산합니다.

- `_run_ingest_dir_job()`
  - 백그라운드에서 인입 작업을 수행하고 상태를 갱신합니다.

- `_is_within_base()`
  - 경로가 `INGEST_BASE_DIR` 안인지 확인합니다.

> 필요하면 `main.py`도 완전한 줄 단위 해설을 추가로 붙여드릴 수 있습니다.

---

## 4) `app/rag/embeddings.py`

1) `import time`
- 재시도 시 대기 시간을 계산하기 위한 모듈입니다.

2) *(빈 줄)*
- 가독성용 줄바꿈입니다.

3) `from google import genai`
- Gemini API 클라이언트를 가져옵니다.

4) `from google.genai import types`
- Gemini API 설정 타입을 가져옵니다.

5) *(빈 줄)*
- 줄바꿈입니다.

6) `from app.core.config import settings`
- 설정을 읽기 위해 임포트합니다.

7) *(빈 줄)*
- 줄바꿈입니다.

8) `client = genai.Client(api_key=settings.gemini_api_key)`
- Gemini API 클라이언트를 생성합니다.

9) *(빈 줄)*
- 줄바꿈입니다.

10) `def embed_texts(...):`
- 여러 텍스트를 임베딩으로 변환하는 함수입니다.

11) `if not texts: return []`
- 입력이 비어 있으면 빈 리스트 반환.

12) `embeddings: list[list[float]] = []`
- 결과를 저장할 리스트를 준비합니다.

13) `for text in texts:`
- 텍스트 하나씩 임베딩 생성.

14) `embeddings.append(_embed_with_retry(...))`
- 실패 시 재시도하는 함수로 임베딩 생성.

15) `return embeddings`
- 결과 반환.

16) *(빈 줄)*
- 줄바꿈입니다.

17) `def _embed_with_retry(...):`
- API 실패 시 재시도 로직을 처리합니다.

18) `max_retries = ...`
- 설정값에서 재시도 횟수 읽기.

19) `backoff = ...`
- 백오프 기본값 읽기.

20) `for attempt in range(max_retries):`
- 재시도 루프 시작.

21) `response = client.models.embed_content(...)`
- Gemini 임베딩 API 호출.

22) `embeddings = response.embeddings or []`
- 응답에서 임베딩 값 추출.

23) `if not embeddings ... raise ValueError(...)`
- 임베딩이 없으면 오류 발생.

24) `return embeddings[0].values`
- 첫 번째 임베딩 벡터 반환.

25) `except Exception: ...`
- 예외가 발생하면 재시도 또는 재발생.

26) `time.sleep(backoff * (2 ** attempt))`
- 재시도 전 대기 (지수 백오프).

---

## 5) `app/rag/vectorstore.py`

이 파일은 **ChromaDB 저장/검색** 전담입니다.

- 컬렉션 생성
- 문서 저장
- 벡터 검색
- source 목록 조회
- DB 초기화

(원하면 줄 단위로 추가 설명 가능합니다)

---

## 6) `app/rag/service.py`

이 파일이 **RAG 핵심 로직**입니다.

- 문서 청킹 + 임베딩 + 저장
- 질의 임베딩 + 검색 + 재정렬 + 답변 생성
- 토큰 카운트
- source 목록
- DB 초기화

(원하면 가장 긴 파일이므로, **완전 줄 단위 버전**을 별도 문서로 만들어 드릴 수 있습니다.)

---

## 7) `app/rag/pdf.py`

1) `from __future__ import annotations`
- 타입 힌트 관련 호환성 개선.

2) `from io import BytesIO`
- 바이트 데이터를 파일처럼 다루기 위한 클래스.

3) `from pypdf import PdfReader`
- PDF 읽기 라이브러리.

4) `def extract_text_by_page(pdf_bytes: bytes) -> list[str]:`
- PDF를 페이지 단위 텍스트 리스트로 변환.

5) `reader = PdfReader(BytesIO(pdf_bytes))`
- PDF 바이트를 PdfReader로 읽음.

6) `pages: list[str] = []`
- 결과 리스트 준비.

7) `for page in reader.pages:`
- 페이지 순회.

8) `text = (page.extract_text() or "").strip()`
- 텍스트 추출 후 공백 제거.

9) `if text: pages.append(text)`
- 텍스트가 있으면 추가.

10) `return pages`
- 결과 반환.

---

## 8) `app/rag/json_ingest.py`

이 파일은 JSON 파일을 문서로 변환합니다.

(문서가 길어 간략 설명만 포함, 필요하면 줄 단위 추가 가능)

---

## 9) `app/rag/directory_ingest.py`

이 파일은 폴더 안의 파일을 읽어 문서로 변환합니다.

(필요하면 줄 단위 설명 확장 가능)

---

## 10) `app/rag/ingest_jobs.py`

이 파일은 비동기 인입 작업 상태를 관리합니다.

(필요하면 줄 단위 설명 확장 가능)

---

### 다음 단계

원하시면 아래 파일들도 **완전 줄 단위**로 확장해드릴 수 있습니다.

- `app/rag/service.py`
- `app/rag/vectorstore.py`
- `app/rag/json_ingest.py`
- `app/rag/directory_ingest.py`
- `app/rag/ingest_jobs.py`

어떤 파일을 우선으로 할지 알려주세요.

