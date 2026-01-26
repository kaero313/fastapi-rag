# FastAPI RAG (Gemini + Chroma)

Gemini 임베딩과 ChromaDB를 이용해 문서를 인입하고, 질의에 대해 근거와 함께
답변하는 FastAPI 기반 RAG 서비스입니다.

## 목적

- 다양한 파일(PDF, JSON, 텍스트)을 빠르게 인입하고 검색 가능한 형태로 저장
- 질의에 대해 근거 문서를 기반으로 한 응답 제공
- 가벼운 UI를 제공해 로컬 환경에서 즉시 사용

## 주요 기능

- 다중 인입 방식 지원
  - 직접 JSON payload 인입 (`/ingest`)
  - PDF 업로드 인입 (`/ingest-pdf`)
  - JSON 업로드 인입 (`/ingest-json`)
  - 디렉터리 일괄 인입 (`/ingest-dir`)
- ChromaDB 영구 저장(로컬 디스크)
- Gemini 임베딩 + LLM 응답 생성
- 쿼리 후보 확장 + 재정렬로 검색 정확도 개선
- 간단한 웹 UI 포함

## 데모

![ui](docs/assets/ui.png)

## 현재 상황

- 핵심 엔드포인트 전체 동작 확인
- 질의 임베딩을 `retrieval_query`로 분리하여 검색 품질 개선
- 대용량 JSON 인입 시 성능 이슈 존재(청크/배치 설정으로 완화)
- google.genai SDK 사용

## 요구사항

- Python 3.10+
- Gemini API Key
- 인터넷 연결 (임베딩/응답 생성용)

## 빠른 시작

### 1) 설치

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 환경 변수

`.env.example`을 참고해 `.env`를 생성합니다.

```env
GEMINI_API_KEY=your_api_key_here
```

### 3) 실행

```bash
uvicorn app.main:app --reload
```

## 데이터 준비

- 기본 인입 폴더: `data/ingest`
- `/ingest-dir`는 `INGEST_BASE_DIR` 하위만 허용됩니다.
- JSON은 `text` 또는 `content` 필드를 인식합니다.

예시 1: 단일 객체

```json
{
  "content": "소아 소화관 중복증은 드문 선천성 질환입니다."
}
```

예시 2: 문서 배열

```json
{
  "documents": [
    {
      "text": "FastAPI는 Python 웹 프레임워크입니다.",
      "metadata": {"source": "docs"}
    }
  ]
}
```

## API 목록

- `GET /` : 웹 UI
- `GET /health` : 헬스체크
- `POST /ingest` : 문서 배열 인입
- `POST /ingest-pdf` : PDF 업로드 인입
- `POST /ingest-json` : JSON 업로드 인입
- `POST /ingest-dir` : 디렉터리 일괄 인입
- `POST /ingest-dir-async` : 디렉터리 비동기 인입 (job 생성)
- `GET /ingest-dir-jobs/{job_id}` : 비동기 인입 상태 조회
- `POST /query` : 질의 및 답변 생성

## 예시

### 문서 인입

```json
{
  "documents": [
    {
      "id": "doc-1",
      "text": "FastAPI is a Python web framework.",
      "metadata": {"source": "docs"}
    },
    {
      "id": "doc-2",
      "text": "RAG combines retrieval with generation."
    }
  ]
}
```

### 질의

```json
{
  "query": "What is RAG?"
}
```

### PDF 업로드

```bash
curl -X POST "http://127.0.0.1:8000/ingest-pdf" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@C:\\path\\to\\document.pdf"
```

### JSON 업로드

```bash
curl -X POST "http://127.0.0.1:8000/ingest-json" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@C:\\path\\to\\data.json"
```

### 디렉터리 인입

```json
{
  "directory": ".",
  "recursive": true,
  "extensions": ["pdf", "json", "txt", "md"]
}
```

`/ingest-dir`는 `INGEST_BASE_DIR` 하위만 허용합니다.
기본값은 `data/ingest`이며, `"."`는 해당 폴더 전체를 의미합니다.

대용량 인입은 `/ingest-dir-async`를 사용하면 요청 타임아웃을 피할 수 있습니다.
비동기 job 상태는 메모리에만 저장되며 서버 재시작 시 초기화됩니다.

## 설정(.env)

주요 변수:

- `GEMINI_API_KEY` (필수)
- `GEMINI_MODEL` (default: `gemini-1.5-flash`)
- `GEMINI_EMBEDDING_MODEL` (default: `text-embedding-004`)
- `EMBED_MAX_RETRIES` (default: `3`)
- `EMBED_RETRY_BACKOFF` (default: `1.0`)
- `INGEST_BATCH_SIZE` (default: `64`)
- `INGEST_CHUNK_SIZE` (default: `4000`)
- `INGEST_CHUNK_OVERLAP` (default: `400`)
- `CANDIDATE_K_MULTIPLIER` (default: `10`)
- `CANDIDATE_K_MIN` (default: `100`)
- `CHROMA_PERSIST_DIR` (default: `data/chroma`)
- `CHROMA_COLLECTION` (default: `rag`)
- `TOP_K` (default: `4`)
- `INGEST_BASE_DIR` (default: `data/ingest`)

## 아키텍처 문서

- `ARCHITECTURE.md` (English)
- `ARCHITECTURE_KO.md` (Korean)

## 트러블슈팅

- 검색 결과가 없거나 엉뚱할 때
  - `top_k` 값을 높여 재시도
  - 인입 폴더/DB 경로가 올바른지 확인
  - 대량 문서는 `INGEST_CHUNK_SIZE` 조정
- 인입이 느릴 때
  - `INGEST_BATCH_SIZE`와 `INGEST_CHUNK_SIZE`를 조정
- Windows에서 `data/chroma` 삭제가 안 될 때
  - 서버/파이썬 프로세스 종료 후 삭제
- 텍스트가 깨질 때
  - JSON 인코딩을 `utf-8-sig` 또는 `cp949`로 저장

## TODO

- [ ] 대용량 인입 비동기/백그라운드 처리
- [ ] 토큰 기반 청킹 적용
- [ ] 메타데이터 필터링 검색 지원
- [ ] 인증/권한 추가
- [ ] 테스트 코드 추가 및 CI 구성
- [ ] Docker/배포 스크립트 제공

## 주의사항

- 인증/권한이 없어 내부망 또는 로컬 환경에서 사용 권장
- 대용량 파일 인입 시 시간이 오래 걸릴 수 있음
- `data/chroma` 삭제 시 벡터 DB 초기화됨 (프로세스 종료 후 삭제)
