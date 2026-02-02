# Beginner Guide: Understanding This RAG Project (Python-Friendly)

이 문서는 **파이썬을 처음 접하는 사람**을 대상으로, 이 프로젝트가 어떻게 동작하는지
코드 구조와 흐름을 아주 자세히 설명합니다. 코드를 읽는 순서와 각 파일의 역할,
데이터가 어떻게 이동하는지도 단계별로 정리했습니다.

---

## 1) 이 프로젝트는 무엇을 하나요?

이 프로젝트는 **RAG(Retrieval-Augmented Generation)** 시스템입니다.

- 사용자가 질문을 하면,
- 관련 문서를 검색해서(벡터 검색),
- 그 문서를 바탕으로 LLM(Gemini)이 답변을 만들어줍니다.

쉽게 말하면:
**"내 문서에서 답을 찾아 요약해주는 챗봇"**입니다.

---

## 2) 폴더 구조 한눈에 보기

```
fastapi-rag/
├─ app/
│  ├─ main.py              # FastAPI 서버 시작점, API 엔드포인트 정의
│  ├─ schemas.py           # 요청/응답 데이터 구조(Pydantic 모델)
│  ├─ core/
│  │  └─ config.py          # 환경변수(.env) 설정 로드
│  ├─ rag/
│  │  ├─ service.py         # RAG 핵심 로직 (인입/검색/답변)
│  │  ├─ embeddings.py      # 임베딩 생성 (Gemini API)
│  │  ├─ vectorstore.py     # ChromaDB 저장/검색
│  │  ├─ pdf.py             # PDF 텍스트 추출
│  │  ├─ json_ingest.py     # JSON 파일 파싱 → 문서화
│  │  ├─ directory_ingest.py# 디렉토리 파일 일괄 인입
│  │  └─ ingest_jobs.py     # 디렉토리 인입 비동기 작업 관리
│  ├─ static/               # 웹 UI (JS/CSS)
│  └─ web/                  # 웹 UI (HTML)
├─ data/
│  ├─ chroma/               # ChromaDB 저장소(벡터 DB)
│  └─ ingest/               # 디렉토리 인입용 기본 폴더
├─ .env                     # 환경변수(로컬에서 직접 생성)
├─ requirements.txt         # 파이썬 패키지 목록
└─ README.md                # 프로젝트 사용 설명서
```

---

## 3) 실행 흐름 요약

1. 서버 실행: `uvicorn app.main:app --reload`
2. 사용자가 `/query` API 호출
3. 질문 임베딩 생성 → Chroma 검색
4. 검색 결과를 프롬프트에 넣고 Gemini에 질문
5. 답변 + 소스 문서 반환

---

## 4) 핵심 파일 설명 (초보자 기준)

### (1) `app/main.py` — **API 입구**

FastAPI로 REST API를 정의합니다.

- `/ingest` : 문서 리스트 직접 인입
- `/ingest-pdf` : PDF 업로드 인입
- `/ingest-json` : JSON 업로드 인입
- `/ingest-dir` : 폴더 전체 인입
- `/sources` : 저장된 문서(source) 목록
- `/query` : 질문 응답
- `/count-tokens` : 토큰 카운트 테스트
- `/reset-db` : DB 초기화

이 파일은 **"사용자가 어떻게 서버와 대화할지"**를 정의합니다.

---

### (2) `app/schemas.py` — **입출력 데이터 구조**

FastAPI는 입력/출력 구조를 Pydantic으로 정의합니다.
예를 들어:

- `QueryRequest`는 `{ "query": "...", "top_k": 4 }` 같은 입력 구조
- `QueryResponse`는 `{ "answer": "...", "sources": [...] }` 같은 응답 구조

즉, **API에서 주고받는 데이터 형태를 명시적으로 정의**합니다.

---

### (3) `app/core/config.py` — **설정(.env) 로드**

환경변수(.env)에 있는 값을 Python에서 읽습니다.

대표 설정:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_EMBEDDING_MODEL`
- `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP`
- `CHROMA_PERSIST_DIR`

여기 값이 바뀌면, 서버를 재시작해야 반영됩니다.

---

### (4) `app/rag/service.py` — **RAG 핵심 로직**

여기가 진짜 핵심입니다.

#### 문서 인입

- 텍스트가 길면 **청킹(chunking)** 해서 잘게 나눔
- 각 청크를 임베딩 생성
- ChromaDB에 저장

#### 질문 처리

1) 질문 임베딩 생성  
2) ChromaDB에서 유사 문서 검색  
3) 문서들을 이어붙여 **Context** 구성  
4) Gemini에 질문 전달 → 답변 생성

---

### (5) `app/rag/embeddings.py` — **임베딩 생성**

Gemini API로 텍스트 임베딩을 생성합니다.

임베딩은 텍스트를 숫자 벡터로 바꾼 것입니다.
이 벡터로 유사도를 계산해 문서를 찾습니다.

---

### (6) `app/rag/vectorstore.py` — **ChromaDB**

ChromaDB는 로컬 벡터 DB입니다.

이 파일에서 하는 일:

- 컬렉션 생성
- 문서/임베딩 저장
- 유사도 검색
- source 목록 조회

즉, **벡터 저장/검색 전담**입니다.

---

### (7) `app/rag/pdf.py` — **PDF 텍스트 추출**

PDF 파일을 읽어서 텍스트만 추출합니다.
추출한 텍스트는 페이지 단위로 쪼개서 인입됩니다.

---

### (8) `app/rag/json_ingest.py` — **JSON 문서화**

JSON 파일을 받아서 `DocumentIn` 형태로 변환합니다.

지원 형태:
- `{"documents": [...]}` 형태
- `[ {...}, ... ]` 배열 형태
- 단일 JSON 객체

---

### (9) `app/rag/directory_ingest.py` — **폴더 인입**

지정한 폴더 안의 파일들을 전부 읽고
PDF/JSON/TXT 등을 자동으로 인입합니다.

---

### (10) `app/rag/ingest_jobs.py` — **비동기 인입**

폴더 인입이 오래 걸릴 수 있으니,
백그라운드 작업으로 실행할 수 있습니다.

---

## 5) 프롬프트가 만들어지는 방식

사용자가 질문하면 다음과 같이 프롬프트를 만들어요:

```
Context:
<검색된 문서 텍스트들>

Question:
<사용자 질문>
```

그리고 시스템 프롬프트는:

```
You are a helpful assistant. Use the provided context to answer the question.
If the answer is not in the context, say you do not know.
```

---

## 6) 자주 묻는 질문 (초보자용)

### Q. 청킹은 왜 하나요?
문서가 너무 길면 검색 품질이 나빠집니다.
잘게 나누면 질문과 관련된 조각을 더 잘 찾을 수 있습니다.

### Q. 벡터DB는 어디에 저장되나요?
`data/chroma/chroma.sqlite3` + UUID 폴더에 저장됩니다.

### Q. 새 설정을 반영하려면?
`.env` 수정 후 서버 재시작 필요합니다.

---

## 7) 추천 학습 순서 (파이썬 초보용)

1. `main.py` — API가 어떻게 생겼는지 보기
2. `schemas.py` — 요청/응답 구조 이해
3. `service.py` — RAG 흐름 이해
4. `vectorstore.py` — 저장/검색 이해
5. `embeddings.py` — 임베딩이 뭔지 이해

---

## 8) 다음 개선 아이디어

- 토큰 기반 청킹을 더 정교하게 만들기
- 메타데이터 필터 확장
- OCR 추가
- 관리자 UI 추가

---

이 문서는 초보자를 위한 **설명서**입니다.
필요하면 각 파일별로 **줄 단위 설명**도 추가할 수 있어요.
