# FastAPI RAG (Gemini + Chroma)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and add your Gemini key.

## Run

```bash
uvicorn app.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /ingest`
- `POST /ingest-pdf`
- `POST /query`

## Example ingest payload

```json
{
  "documents": [
    {"id": "doc-1", "text": "FastAPI is a Python web framework.", "metadata": {"source": "docs"}},
    {"id": "doc-2", "text": "RAG combines retrieval with generation."}
  ]
}
```

## Example query payload

```json
{
  "query": "What is RAG?"
}
```

## Example PDF upload

```bash
curl -X POST "http://127.0.0.1:8000/ingest-pdf" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@C:\\path\\to\\document.pdf"
```
