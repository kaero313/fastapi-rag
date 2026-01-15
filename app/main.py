from fastapi import FastAPI

from app.rag.service import answer_query, ingest_documents
from app.schemas import IngestRequest, QueryRequest, QueryResponse

app = FastAPI(title="fastapi-rag")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    ids = ingest_documents(request.documents)
    return {"ingested": len(ids), "ids": ids}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return answer_query(request.query, top_k=request.top_k)
