from fastapi import FastAPI, File, HTTPException, UploadFile

from app.rag.service import answer_query, ingest_documents, ingest_pdf_bytes
from app.schemas import IngestRequest, QueryRequest, QueryResponse

app = FastAPI(title="fastapi-rag")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    ids = ingest_documents(request.documents)
    return {"ingested": len(ids), "ids": ids}


@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty PDF file.")
    ids = ingest_pdf_bytes(file.filename, content)
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF.",
        )
    return {"ingested": len(ids), "ids": ids}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return answer_query(request.query, top_k=request.top_k)
