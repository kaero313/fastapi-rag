# FastAPI RAG Architecture

This document describes the API surface, internal modules, and end-to-end
data flow for the FastAPI RAG system (Gemini + Chroma).

## At a glance

- Backend: FastAPI app in `app/main.py` served by Uvicorn.
- RAG core: `app/rag/service.py` orchestrates ingestion, retrieval, and generation.
- Embeddings: Gemini embedding model via `google.generativeai`.
- Vector store: ChromaDB persistent collection on disk.
- Frontend: static HTML/CSS/JS served by FastAPI.

## System diagram (high level)

```
             +-------------------------+
             |      Browser UI         |
             |  app/web + app/static   |
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

## API endpoints

All endpoints are defined in `app/main.py`.

### GET /

Serves the UI.

- Response: `app/web/index.html`

### GET /health

Health check.

- Response: `{ "status": "ok" }`

### POST /ingest

Ingest a list of documents directly.

- Request body: `IngestRequest`
  - `documents`: list of `DocumentIn`
- Response: `{ "ingested": <int>, "ids": [<str>, ...] }`
- Errors: standard FastAPI validation errors

### POST /ingest-pdf

Upload a PDF file and ingest extracted text by page.

- Request: multipart form with `file`
- Response: `{ "ingested": <int>, "ids": [<str>, ...] }`
- Errors:
  - 400 if file is missing or not `.pdf`
  - 400 if PDF has no extractable text

### POST /ingest-json

Upload a JSON file and ingest parsed records.

- Request: multipart form with `file`
- Response: `{ "ingested": <int>, "ids": [<str>, ...] }`
- Errors:
  - 400 if file is missing or not `.json`
  - 400 if JSON is invalid
  - 400 if JSON yields no ingestable records

### POST /ingest-dir

Ingest files from a directory under `INGEST_BASE_DIR`.

- Request body: `IngestDirectoryRequest`
  - `directory`: relative or absolute path
  - `recursive`: default true
  - `extensions`: optional list, e.g. `["pdf", "json"]`
- Response:
  - `{ "ingested": <int>, "ids": [...], "files_processed": <int>, ... }`
  - Optional `error_files` list (up to 5 entries)
- Errors:
  - 400 if directory is outside `INGEST_BASE_DIR`
  - 404 if target directory does not exist
  - 400 if no ingestable files found

### POST /query

Query the vector store and generate a response.

- Request body: `QueryRequest`
  - `query`: string
  - `top_k`: optional integer
- Response: `QueryResponse`
  - `answer`: string
  - `sources`: list of sources with `id`, `score`, `text`, `metadata`

## Data models (schemas)

Defined in `app/schemas.py`.

- `DocumentIn`
  - `id`: optional string
  - `text`: string
  - `metadata`: optional object
- `IngestRequest`
  - `documents`: list of `DocumentIn`
- `IngestDirectoryRequest`
  - `directory`: string
  - `recursive`: bool (default true)
  - `extensions`: optional list of strings
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

## Core workflow details

### Ingestion flow

1. Input is converted to a list of `DocumentIn`.
2. Text is chunked if longer than `INGEST_CHUNK_SIZE`.
3. Documents are embedded using Gemini embeddings with task type
   `retrieval_document`.
4. Vectors and documents are stored in ChromaDB with metadata.

Key behaviors in `app/rag/service.py`:

- If any document has metadata, all documents are stored with metadata.
  Missing metadata uses `{"_missing": True}`.
- IDs are generated with `uuid4().hex` when not provided.
- Embedding calls are batched by `INGEST_BATCH_SIZE`.
- Long documents are split into multiple `DocumentIn` records.

### Directory ingestion flow

Implemented in `app/rag/directory_ingest.py`.

- Supports extensions:
  - `.pdf`, `.json`, `.txt`, `.md`, `.markdown`, `.csv`, `.log`
- If `extensions` are provided, they override the built-in list.
- Uses `Path.glob("**/*")` for recursive enumeration.
- Per-file handling:
  - PDF: page-by-page extraction via `pypdf`.
  - JSON: parsed via `parse_json_documents`.
  - Text-like: UTF-8 read with `errors="replace"`.
- Returns `stats` with processed, skipped, failed, and empty file counts.

### JSON ingestion rules

Implemented in `app/rag/json_ingest.py`.

- Tries `utf-8-sig`, then falls back to `cp949`.
- Accepts:
  - `{ "documents": [ ... ] }`
  - a list of items
  - a single object or value
- Per-item handling:
  - If dict has `text`, use it (and optional `metadata`).
  - If dict has `content`, use it as text.
  - Otherwise serialize the item as JSON.

### PDF ingestion rules

Implemented in `app/rag/pdf.py`.

- Uses `pypdf.PdfReader` and `page.extract_text()`.
- Each non-empty page becomes a separate `DocumentIn`.

### Query flow

1. Embed the query using task type `retrieval_query`.
2. Retrieve candidate results from ChromaDB
   (`candidate_k = max(top_k * 5, 50)`).
3. Rerank candidates with a lexical match score, then by distance.
4. Build context from the top `top_k` chunks.
5. Call Gemini LLM to generate the final answer.

Reranking is defined in `app/rag/service.py` as `_rerank_results`.

## Vector store details

Defined in `app/rag/vectorstore.py`.

- Persistent storage uses `chromadb.PersistentClient`.
- Collection name from `CHROMA_COLLECTION`.
- Vector distance is cosine (`metadata={"hnsw:space": "cosine"}`).
- Stored fields: `ids`, `documents`, `embeddings`, optional `metadatas`.

## Embedding and generation

Defined in `app/rag/embeddings.py` and `app/rag/service.py`.

- Embedding model: `GEMINI_EMBEDDING_MODEL`
- LLM model: `GEMINI_MODEL`
- Embedding retry:
  - `EMBED_MAX_RETRIES`
  - exponential backoff with `EMBED_RETRY_BACKOFF`
- Query uses `retrieval_query`, documents use `retrieval_document`.

## Frontend behavior

Files:

- `app/web/index.html`
- `app/static/app.js`
- `app/static/styles.css`

Key behaviors:

- On load, calls `GET /health` and updates status indicator.
- Sends `POST /query` with `query` and `top_k` from input.
- Renders up to 4 sources in the UI.
- `top_k` input is constrained in HTML to `[1, 12]`.

## Configuration

Settings are loaded from `.env` by Pydantic settings in
`app/core/config.py`.

Required:

- `GEMINI_API_KEY`

Optional (defaults shown):

- `GEMINI_MODEL=gemini-1.5-flash`
- `GEMINI_EMBEDDING_MODEL=text-embedding-004`
- `EMBED_MAX_RETRIES=3`
- `EMBED_RETRY_BACKOFF=1.0`
- `INGEST_BATCH_SIZE=64`
- `INGEST_CHUNK_SIZE=4000`
- `CHROMA_PERSIST_DIR=data/chroma`
- `CHROMA_COLLECTION=rag`
- `TOP_K=4`
- `INGEST_BASE_DIR=data/ingest`

## File layout (key parts)

```
app/
  main.py                 FastAPI app and routes
  schemas.py              Pydantic request/response models
  core/config.py          Environment configuration
  rag/
    service.py            RAG orchestration (ingest, query, rerank)
    embeddings.py         Gemini embedding wrapper with retries
    vectorstore.py        ChromaDB access layer
    directory_ingest.py   File system ingestion helpers
    json_ingest.py        JSON parsing into DocumentIn
    pdf.py                PDF text extraction
  static/                 Frontend JS/CSS
  web/                    Frontend HTML
data/
  ingest/                 Default ingest directory
  chroma/                 Chroma persistent store
```

## Operational notes

- No authentication or authorization is implemented.
- Ingestion and query are synchronous and can be slow for large inputs.
- The system relies on external Gemini APIs for embeddings and generation.
- Large documents are split by character count, not tokens.
