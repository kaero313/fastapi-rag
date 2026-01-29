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
    source: str | None = None
    sources: list[str] | None = None
    page_gte: int | None = None
    page_lte: int | None = None


class IngestDirectoryRequest(BaseModel):
    directory: str
    recursive: bool = True
    extensions: list[str] | None = None


class CountTokensRequest(BaseModel):
    text: str
    model: str | None = None


class Source(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class CountTokensResponse(BaseModel):
    model: str
    tokens: int


class SourcesResponse(BaseModel):
    sources: list[str]
