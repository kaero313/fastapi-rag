# Codex Context (Project State)

Purpose:
- This file is for future Codex sessions. Read it first to restore context.

Project:
- Name: fastapi-rag
- Stack: FastAPI + Gemini (google.generativeai) + ChromaDB
- OS/Shell used last: Windows, PowerShell
- Repo root: C:\project\fastapi-rag

Key docs:
- ARCHITECTURE.md (English system overview)
- ARCHITECTURE_KO.md (Korean system overview)

Current runtime commands:
- Start server: `uvicorn app.main:app --reload`
- Health check: `GET http://127.0.0.1:8000/health`

API surface (defined in app/main.py):
- GET /
- GET /health
- POST /ingest
- POST /ingest-pdf
- POST /ingest-json
- POST /ingest-dir
- POST /query

Important config (app/core/config.py):
- GEMINI_API_KEY (required, in .env, do not commit)
- GEMINI_MODEL (default gemini-1.5-flash)
- GEMINI_EMBEDDING_MODEL (default text-embedding-004)
- EMBED_MAX_RETRIES (default 3)
- EMBED_RETRY_BACKOFF (default 1.0)
- INGEST_BATCH_SIZE (default 64)
- INGEST_CHUNK_SIZE (default 4000; .env currently sets 8000)
- CHROMA_PERSIST_DIR (default data/chroma)
- CHROMA_COLLECTION (default rag)
- TOP_K (default 4)
- INGEST_BASE_DIR (default data/ingest)

Data/DB state at last check:
- Chroma collection count (rag): 100
  - Check with:
    `python -c "import chromadb; from app.core.config import settings; c=chromadb.PersistentClient(path=settings.chroma_persist_dir).get_or_create_collection(name=settings.chroma_collection); print('count', c.count())"`
- data/ingest contains many JSON files; some are very large.
  - Example huge file: data/ingest/cid_54095_1_5752.json (content ~18M chars).
- data/chroma is ignored by git; it can be reset safely after stopping processes.

Behavior changes in code (already in repo state):
- app/rag/embeddings.py
  - embed_texts now accepts task_type; query uses retrieval_query.
  - retry with exponential backoff via EMBED_MAX_RETRIES and EMBED_RETRY_BACKOFF.
- app/rag/service.py
  - ingestion batches via INGEST_BATCH_SIZE.
  - document chunking via INGEST_CHUNK_SIZE (character count).
  - query uses candidate_k = max(top_k * 5, 50) and reranks.
  - rerank prefers lexical matches, then vector distance.
- app/rag/json_ingest.py
  - supports `content` field in JSON records.
  - decodes utf-8-sig, falls back to cp949.
- .env.example updated with EMBED_* and INGEST_* keys.
- .env currently has INGEST_CHUNK_SIZE=8000 (do not commit .env).

Known issues and constraints:
- Embedding API payload limit ~4MB.
  - Chunking avoids this, but too-large INGEST_CHUNK_SIZE can still fail.
- Ingestion can be very slow with large JSON content.
  - 8000 char chunks reduce count but still heavy.
- Chroma file locks on Windows:
  - Stop uvicorn and any Python process before deleting data/chroma.
  - If delete fails, find/stop python.exe holding locks, then retry.
- google.generativeai emits deprecation warnings (package is deprecated).

Why queries sometimes looked empty or irrelevant:
- Previously, query embeddings used retrieval_document task type.
  - Now fixed to retrieval_query.
- Top-k in UI is limited to 12 (app/web/index.html).
  - Retrieval now fetches larger candidate set and reranks.

Quick validation (use UTF-8 safe requests):
- Query example string (Korean):
  - \uc18c\uc544 \uc18c\ud654\uad00 \uc911\ubcf5\uc99d
- Test via Python (avoids PowerShell encoding issues):
  - python -c "import requests, json; payload={'query':'\\uc18c\\uc544 \\uc18c\\ud654\\uad00 \\uc911\\ubcf5\\uc99d','top_k':12}; r=requests.post('http://127.0.0.1:8000/query', json=payload, timeout=30); print(r.status_code); print(r.json().get('sources',[{}])[0].get('metadata',{}).get('source'))"
  - Expected top source: cid_1549_1_4872.json

If DB needs a clean reset:
1) Stop uvicorn and any python process using Chroma.
2) Delete data/chroma.
3) Re-run /ingest-dir.

Notes for future Codex:
- Prefer reading ARCHITECTURE.md first for system overview.
- Avoid printing .env contents (contains real API key).
- Use Python -c or requests for tests to avoid PowerShell encoding pitfalls.
