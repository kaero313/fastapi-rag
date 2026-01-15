from typing import Any

from pydantic import BaseModel


class DocumentIn(BaseModel):
    id: str | None = None
    text: str
    metadata: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    documents: list[DocumentIn]


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class Source(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
