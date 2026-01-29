# FastAPI RAG 아키텍처 (한국어)

이 문서는 FastAPI RAG 시스템의 API 목록, 내부 모듈, 데이터 흐름, 설정을
한눈에 볼 수 있도록 정리한 상세 설명서입니다.

## 한눈에 보기

- 백엔드: FastAPI (`app/main.py`) + Uvicorn
- RAG 핵심: `app/rag/service.py`가 ingest/query/generate를 오케스트레이션
- 임베딩: Gemini 임베딩 API (`google.genai`)
- 벡터 DB: ChromaDB (영구 저장)
- 프론트엔드: 정적 HTML/CSS/JS (FastAPI로 서빙)

## 시스템 다이어그램 (요약)

```
             +-------------------------+
             |        Browser UI       |
             |   app/web + app/static  |
             +------------+------------+
                          |
                          | HTTP
                          v
                 +--------+--------+
                 |   FastAPI app   |
                 |   app/main.py   |
                 +--------+--------+
                          |
         +----------------+----------------+
         |                                 |
         v                                 v
  +------+--------+                 +------+--------+
  | Ingestion     |                 | Query/Answer  |
  | service       |                 | service       |
  | rag/service.py|                 | rag/service.py|
  +------+--------+                 +------+--------+
         |                                 |
         | embed texts                     | embed query
         v                                 v
  +------+--------+                 +------+--------+
  | Gemini Embed  |                 | Gemini Embed  |
  | (documents)   |                 | (query)       |
  +------+--------+                 +------+--------+
         |                                 |
         | store vectors                   | vector search
         v                                 v
   +-----+-------------------------------+-----+
   |           ChromaDB (persistent)           |
   |      data/chroma, collection "rag"        |
   +-----+-------------------------------+-----+
                          |
                          | context chunks
                          v
                 +--------+--------+
                 | Gemini LLM      |
                 | (answer)        |
                 +-----------------+
```

## API 엔드포인트

모든 엔드포인트는 `app/main.py`에 정의되어 있습니다.

### GET /

UI 서빙.

- Response: `app/web/index.html`

### GET /health

헬스체크.

- Response: `{ "status": "ok" }`

### POST /ingest

문서 배열 직접 인입.

- Request body: `IngestRequest`
  - `documents`: `DocumentIn` 배열
- Response: `{ "ingested": <int>, "ids": [<str>, ...] }`
- Errors: FastAPI 검증 오류

### POST /ingest-pdf  

PDF 업로드 후 페이지별 텍스트 인입.

- Request: multipart form `file`
- Response: `{ "ingested": <int>, "ids": [<str>, ...] }`
- Errors:
  - 400: 파일 없음 또는 `.pdf` 아님
  - 400: 텍스트 추출 불가

### POST /ingest-json

JSON 업로드 후 레코드 인입.

- Request: multipart form `file`
- Response: `{ "ingested": <int>, "ids": [<str>, ...] }`
- Errors:
  - 400: 파일 없음 또는 `.json` 아님
  - 400: JSON 파싱 실패
  - 400: 인입 가능한 레코드 없음

### POST /ingest-dir

디렉터리 기반 인입 (`INGEST_BASE_DIR` 하위).

- Request body: `IngestDirectoryRequest`
  - `directory`: 상대/절대 경로
  - `recursive`: 기본 true
  - `extensions`: 선택, 예 `["pdf", "json"]`
- Response:
  - `{ "ingested": <int>, "ids": [...], "files_processed": <int>, ... }`
  - `error_files` 최대 5개
- Errors:
  - 400: `INGEST_BASE_DIR` 밖 경로
  - 404: 디렉터리 없음
  - 400: 인입 가능한 파일 없음

### POST /ingest-dir-async

디렉터리 인입을 백그라운드 스레드에서 실행하고 job ID를 반환합니다.

- Request body: `IngestDirectoryRequest`
- Response: `{ "job_id": "<id>", "status": "queued" }`
- Errors: `/ingest-dir`와 동일

### GET /ingest-dir-jobs/{job_id}

백그라운드 인입 job 상태를 조회합니다.

- Response fields:
  - `job_id`, `status`, `created_at`, `started_at`, `finished_at`
  - 실패 시 `error`
  - 완료 시 `result` (`/ingest-dir`와 동일한 결과)
- Errors:
  - 404: job 없음

### POST /query

벡터 검색 후 답변 생성.

- Request body: `QueryRequest`
  - `query`: 문자열
  - `top_k`: 선택
- Response: `QueryResponse`
  - `answer`: 문자열
  - `sources`: `id`, `score`, `text`, `metadata`

## 데이터 모델

`app/schemas.py`에 정의.

- `DocumentIn`
  - `id`: optional string
  - `text`: string
  - `metadata`: optional object
- `IngestRequest`
  - `documents`: list of `DocumentIn`
- `IngestDirectoryRequest`
  - `directory`: string
  - `recursive`: bool
  - `extensions`: optional list
- `QueryRequest`
  - `query`: string
  - `top_k`: optional int
- `Source`
  - `id`: string
  - `score`: float
  - `text`: string
  - `metadata`: optional object
- `QueryResponse`
  - `answer`: string
  - `sources`: list of `Source`

## 핵심 동작 흐름

### 인입(ingest)

1. 입력을 `DocumentIn` 배열로 변환.
2. `INGEST_CHUNK_SIZE` 기준으로 길이를 분할.
3. Gemini 임베딩(`retrieval_document`) 생성.
4. ChromaDB에 벡터 + 메타데이터 저장.

`app/rag/service.py` 주요 규칙:

- 하나라도 metadata가 있으면 전체를 metadata와 함께 저장.
- `id` 없으면 `uuid4().hex` 생성.
- `INGEST_BATCH_SIZE` 단위로 임베딩 요청.
- 문서 길이가 크면 여러 조각으로 분할.

### 디렉터리 인입

`app/rag/directory_ingest.py`.

- 지원 확장자:
  - `.pdf`, `.json`, `.txt`, `.md`, `.markdown`, `.csv`, `.log`
- `extensions`가 주어지면 그 값만 사용.
- 재귀 탐색: `Path.glob("**/*")`.
- 파일별 처리:
  - PDF: 페이지별 텍스트 추출
  - JSON: `parse_json_documents`
  - 텍스트: UTF-8, 오류는 replace
- `stats`에 처리/스킵/실패/빈 파일 수 기록.

### JSON 파싱 규칙

`app/rag/json_ingest.py`.

- `utf-8-sig` 우선, 실패 시 `cp949`로 디코딩.
- 허용 입력:
  - `{ "documents": [ ... ] }`
  - 리스트
  - 단일 오브젝트/값
- 레코드 처리:
  - `text` 키가 있으면 사용
  - `content` 키가 있으면 사용
  - 그 외는 JSON 문자열로 직렬화

### PDF 파싱 규칙

`app/rag/pdf.py`.

- `pypdf.PdfReader` 사용.
- 텍스트가 있는 페이지만 인입.

### 질의(query)

1. 질의 임베딩 생성 (`retrieval_query`).
2. 후보군 조회 (`candidate_k = max(top_k * CANDIDATE_K_MULTIPLIER, CANDIDATE_K_MIN)`).
3. 키워드 매칭 기반 재정렬 후 상위 `top_k` 선택.
4. 컨텍스트를 묶어 Gemini LLM 호출.

재정렬 함수는 `app/rag/service.py`의 `_rerank_results`.

## 벡터 스토어

`app/rag/vectorstore.py`.

- `chromadb.PersistentClient` 사용.
- 컬렉션 이름: `CHROMA_COLLECTION`.
- 코사인 거리 (`hnsw:space = cosine`).
- 저장 필드: `ids`, `documents`, `embeddings`, `metadatas`.

## 임베딩/생성

`app/rag/embeddings.py`, `app/rag/service.py`.

- 임베딩 모델: `GEMINI_EMBEDDING_MODEL`
- LLM 모델: `GEMINI_MODEL`
- 재시도:
  - `EMBED_MAX_RETRIES`
  - `EMBED_RETRY_BACKOFF` (지수 백오프)
- 문서와 질의는 task_type을 분리
  - 문서: `retrieval_document`
  - 질의: `retrieval_query`

## 프론트엔드 동작

파일:

- `app/web/index.html`
- `app/static/app.js`
- `app/static/styles.css`

기능 요약:

- 로딩 시 `GET /health` 호출
- `POST /query`로 질의 전송
- 최대 4개의 source 표시
- `top_k` 입력은 HTML에서 1~12로 제한

## 설정

`app/core/config.py`에서 `.env`를 로드.

필수:

- `GEMINI_API_KEY`

기본값(선택):

- `GEMINI_MODEL=gemini-2.5-flash`
- `GEMINI_EMBEDDING_MODEL=text-embedding-004`
- `EMBED_MAX_RETRIES=3`
- `EMBED_RETRY_BACKOFF=1.0`
- `INGEST_BATCH_SIZE=64`
- `INGEST_CHUNK_SIZE=4000`
- `CANDIDATE_K_MULTIPLIER=10`
- `CANDIDATE_K_MIN=100`
- `CHROMA_PERSIST_DIR=data/chroma`
- `CHROMA_COLLECTION=rag`
- `TOP_K=4`
- `INGEST_BASE_DIR=data/ingest`

## 파일 구조

```
app/
  main.py                 FastAPI 라우터
  schemas.py              요청/응답 모델
  core/config.py          환경 설정
  rag/
    service.py            인입/검색/생성 오케스트레이션
    embeddings.py         Gemini 임베딩 래퍼
    vectorstore.py        ChromaDB 접근
    directory_ingest.py   디렉터리 인입 유틸
    json_ingest.py        JSON 파싱
    pdf.py                PDF 텍스트 추출
  static/                 프론트엔드 JS/CSS
  web/                    프론트엔드 HTML
data/
  ingest/                 인입 대상 디렉터리
  chroma/                 ChromaDB 저장소
```

## 운영 참고사항

- 인증/권한 없음.
- 대량 인입 시 동기식 처리로 시간이 오래 걸릴 수 있음.
- 비동기 인입 job 상태는 메모리에만 저장되며 서버 재시작 시 초기화됨.
- Gemini API 의존.
- 문서 분할은 토큰이 아닌 문자 수 기준.
