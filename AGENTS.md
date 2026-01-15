# Project Summary: fastapi-rag

## Purpose
- FastAPI-based RAG service using Gemini for embeddings + chat completions and ChromaDB for persistent vector storage.

## Runtime
- App entry: `app/main.py` with endpoints `/health`, `/ingest`, `/query`.
- Start: `uvicorn app.main:app --reload`.

## Data flow
- Ingest: `/ingest` -> `app.rag.service.ingest_documents` -> embed texts with Gemini -> store in Chroma collection.
- Query: `/query` -> embed query -> Chroma similarity search -> build context -> Gemini chat completion -> return answer + sources.

## Config
- `app/core/config.py` (pydantic-settings) reads `.env`.
- Key env vars: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`, `TOP_K`.
- Defaults: model `gemini-1.5-flash`, embedding `text-embedding-004`, persist `data/chroma`, collection `rag`, top_k `4`.

## Storage
- Chroma persistent client at `data/chroma` (relative path by default).
- Collection configured with cosine space.

## Key files
- `app/main.py`: FastAPI routes.
- `app/schemas.py`: request/response models.
- `app/rag/service.py`: ingest/query logic and Gemini chat.
- `app/rag/embeddings.py`: Gemini embeddings.
- `app/rag/vectorstore.py`: Chroma persistence and search.
- `requirements.txt`: fastapi, uvicorn, pydantic, openai, chromadb.

## Notes
- Query scoring uses `1.0 - distance` (cosine distance).
- If context does not contain answer, system prompt asks model to say "I do not know".
