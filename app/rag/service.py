from __future__ import annotations

from pathlib import Path
from typing import Iterable
from uuid import uuid4

from google import genai
from google.genai import types

from app.core.config import settings
from app.rag.directory_ingest import load_documents_from_directory
from app.rag.embeddings import embed_texts
from app.rag.json_ingest import parse_json_documents
from app.rag.pdf import extract_text_by_page
from app.rag.vectorstore import add_documents, query_by_embedding
from app.schemas import DocumentIn, QueryResponse, Source

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "If the answer is not in the context, say you do not know."
)

client = genai.Client(api_key=settings.gemini_api_key)


def ingest_documents(documents: list[DocumentIn]) -> list[str]:
    ids: list[str] = []
    include_metadata = any(doc.metadata for doc in documents)
    chunk_size = max(1, settings.ingest_chunk_size)
    expanded_documents = _split_documents(documents, chunk_size)

    batch_size = max(1, settings.ingest_batch_size)
    for batch in _chunked(expanded_documents, batch_size):
        texts: list[str] = []
        metadatas: list[dict[str, object]] = []
        batch_ids: list[str] = []

        for doc in batch:
            doc_id = doc.id or uuid4().hex
            batch_ids.append(doc_id)
            texts.append(doc.text)
            if include_metadata:
                metadata = doc.metadata or {"_missing": True}
                metadatas.append(metadata)

        embeddings = embed_texts(texts, task_type="retrieval_document")
        add_documents(
            ids=batch_ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas if include_metadata else None,
        )
        ids.extend(batch_ids)
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


def ingest_json_bytes(filename: str, json_bytes: bytes) -> list[str]:
    documents = parse_json_documents(filename, json_bytes)
    if not documents:
        return []
    return ingest_documents(documents)


def ingest_directory(
    directory: Path,
    recursive: bool = True,
    extensions: list[str] | None = None,
    base_dir: Path | None = None,
) -> tuple[list[str], dict[str, object]]:
    documents, stats = load_documents_from_directory(
        directory,
        recursive=recursive,
        extensions=extensions,
        base_dir=base_dir,
    )
    if not documents:
        return [], stats
    ids = ingest_documents(documents)
    return ids, stats


def answer_query(query: str, top_k: int | None = None) -> QueryResponse:
    effective_top_k = top_k or settings.top_k
    query_embedding = embed_texts([query], task_type="retrieval_query")[0]
    candidate_k = max(effective_top_k * 5, 50)
    results = query_by_embedding(query_embedding, top_k=candidate_k)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    sources: list[Source] = []
    context_chunks: list[str] = []

    ranked = _rerank_results(query, ids, documents, metadatas, distances)
    for doc_id, text, metadata, distance in ranked[:effective_top_k]:
        score = 1.0 - float(distance) if distance is not None else 0.0
        sources.append(
            Source(id=doc_id, score=score, text=text, metadata=metadata)
        )
        context_chunks.append(text)

    context = "\n\n".join(context_chunks)

    prompt = "Context:\n" + context + "\n\n" + "Question:\n" + query
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    answer = response.text or ""
    return QueryResponse(answer=answer, sources=sources)


def _chunked(items: list[DocumentIn], size: int) -> Iterable[list[DocumentIn]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _split_documents(
    documents: list[DocumentIn],
    chunk_size: int,
) -> list[DocumentIn]:
    if chunk_size <= 0:
        return documents

    chunked: list[DocumentIn] = []
    for doc in documents:
        text = doc.text
        if len(text) <= chunk_size:
            chunked.append(doc)
            continue

        for chunk in _chunk_text(text, chunk_size):
            chunked.append(
                DocumentIn(
                    text=chunk,
                    metadata=doc.metadata,
                )
            )
    return chunked


def _chunk_text(text: str, chunk_size: int) -> Iterable[str]:
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def _rerank_results(
    query: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, object] | None],
    distances: list[float | None],
) -> list[tuple[str, str, dict[str, object] | None, float | None]]:
    query_lower = query.lower()
    terms = [term for term in query_lower.split() if term]

    def lexical_score(text: str) -> int:
        if not text:
            return 0
        lowered = text.lower()
        score = 0
        if query_lower and query_lower in lowered:
            score += len(terms) + 1
        for term in terms:
            if term in lowered:
                score += 1
        return score

    scored: list[tuple[int, float, int, str, str, dict[str, object] | None, float | None]] = []
    for index, (doc_id, text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances)
    ):
        score = lexical_score(text)
        dist_value = float(distance) if distance is not None else 1.0
        scored.append((score, dist_value, index, doc_id, text, metadata, distance))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(doc_id, text, metadata, distance) for _, _, _, doc_id, text, metadata, distance in scored]
