# Project Summary: fastapi-rag

## Purpose
- FastAPI-based RAG service using Gemini for embeddings + chat completions and ChromaDB for persistent vector storage.

## Runtime
- App entry: `app/main.py` with endpoints `/health`, `/ingest`, `/ingest-pdf`, `/ingest-json`, `/ingest-dir`, `/query`.
- Start: `uvicorn app.main:app --reload`.

## Data flow
- Ingest: `/ingest` -> chunk text with overlap -> embed texts with Gemini -> store in Chroma collection.
- PDF ingest: `/ingest-pdf` -> extract text per page -> embed texts with Gemini -> store in Chroma collection.
- JSON ingest: `/ingest-json` -> parse JSON to documents -> embed texts with Gemini -> store in Chroma collection.
- Directory ingest: `/ingest-dir` -> scan files in `INGEST_BASE_DIR` -> parse (pdf/json/text) -> embed -> store.
- Query: `/query` -> embed query -> Chroma similarity search -> build context -> Gemini chat completion -> return answer + sources.

## Config
- `app/core/config.py` (pydantic-settings) reads `.env`.
- Key env vars: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, `EMBED_MAX_RETRIES`, `EMBED_RETRY_BACKOFF`, `INGEST_BATCH_SIZE`, `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP`, `CANDIDATE_K_MULTIPLIER`, `CANDIDATE_K_MIN`, `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`, `TOP_K`, `INGEST_BASE_DIR`.
- Defaults: model `gemini-1.5-flash`, embedding `text-embedding-004`, chunk size `4000`, overlap `400`, persist `data/chroma`, collection `rag`, top_k `4`, ingest base `data/ingest`.

## Storage
- Chroma persistent client at `data/chroma` (relative path by default).
- Collection configured with cosine space.

## Key files
- `app/main.py`: FastAPI routes.
- `app/schemas.py`: request/response models.
- `app/rag/service.py`: ingest/query logic and Gemini chat.
- `app/rag/embeddings.py`: Gemini embeddings.
- `app/rag/pdf.py`: PDF text extraction.
- `app/rag/json_ingest.py`: JSON parsing for ingest.
- `app/rag/directory_ingest.py`: Directory scan + file parsing.
- `app/rag/vectorstore.py`: Chroma persistence and search.
- `requirements.txt`: fastapi, uvicorn, pydantic, google-genai, chromadb, pypdf, python-multipart.

## Notes
- Query scoring uses `1.0 - distance` (cosine distance).
- If context does not contain answer, system prompt asks model to say "I do not know".
