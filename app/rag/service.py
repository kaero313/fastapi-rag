from __future__ import annotations

from uuid import uuid4

import google.generativeai as genai

from app.core.config import settings
from app.rag.embeddings import embed_texts
from app.rag.pdf import extract_text_by_page
from app.rag.vectorstore import add_documents, query_by_embedding
from app.schemas import DocumentIn, QueryResponse, Source

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "If the answer is not in the context, say you do not know."
)

genai.configure(api_key=settings.gemini_api_key)


def ingest_documents(documents: list[DocumentIn]) -> list[str]:
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, object]] = []
    include_metadata = any(doc.metadata for doc in documents)

    for doc in documents:
        doc_id = doc.id or uuid4().hex
        ids.append(doc_id)
        texts.append(doc.text)
        if include_metadata:
            metadata = doc.metadata or {"_missing": True}
            metadatas.append(metadata)

    embeddings = embed_texts(texts)
    add_documents(
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas if include_metadata else None,
    )
    return ids


def ingest_pdf_bytes(filename: str, pdf_bytes: bytes) -> list[str]:
    pages = extract_text_by_page(pdf_bytes)
    documents: list[DocumentIn] = []
    for index, text in enumerate(pages, start=1):
        documents.append(
            DocumentIn(
                text=text,
                metadata={"source": filename, "page": index},
            )
        )
    if not documents:
        return []
    return ingest_documents(documents)


def answer_query(query: str, top_k: int | None = None) -> QueryResponse:
    effective_top_k = top_k or settings.top_k
    query_embedding = embed_texts([query])[0]
    results = query_by_embedding(query_embedding, top_k=effective_top_k)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    sources: list[Source] = []
    context_chunks: list[str] = []

    for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        score = 1.0 - float(distance) if distance is not None else 0.0
        sources.append(
            Source(id=doc_id, score=score, text=text, metadata=metadata)
        )
        context_chunks.append(text)

    context = "\n\n".join(context_chunks)

    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=SYSTEM_PROMPT,
    )
    prompt = "Context:\n" + context + "\n\n" + "Question:\n" + query
    response = model.generate_content(prompt)
    answer = response.text or ""
    return QueryResponse(answer=answer, sources=sources)
