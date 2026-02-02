from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
from uuid import uuid4

from google import genai
from google.genai import types

from app.core.config import settings
from app.rag.directory_ingest import load_documents_from_directory
from app.rag.embeddings import embed_texts
from app.rag.json_ingest import parse_json_documents
from app.rag.pdf import extract_text_by_page
from app.rag.vectorstore import (
    add_documents,
    list_sources,
    query_by_embedding,
    reset_store,
)
from app.schemas import DocumentIn, QueryResponse, Source

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "If the answer is not in the context, say you do not know."
)

client = genai.Client(api_key=settings.gemini_api_key)


def ingest_documents(documents: list[DocumentIn]) -> list[str]:
    ids: list[str] = []
    chunk_tokens = max(0, settings.ingest_chunk_size)
    chunk_overlap = max(0, settings.ingest_chunk_overlap)
    expanded_documents = _split_documents(documents, chunk_tokens, chunk_overlap)
    include_metadata = any(doc.metadata for doc in expanded_documents)

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


def answer_query(
    query: str,
    top_k: int | None = None,
    source: str | None = None,
    sources: list[str] | None = None,
    page_gte: int | None = None,
    page_lte: int | None = None,
) -> QueryResponse:
    effective_top_k = top_k or settings.top_k
    query_embedding = embed_texts([query], task_type="retrieval_query")[0]
    multiplier = max(1, settings.candidate_k_multiplier)
    candidate_k = max(
        effective_top_k * multiplier,
        settings.candidate_k_min,
    )
    where = _build_where(source, sources, page_gte, page_lte)
    results = query_by_embedding(query_embedding, top_k=candidate_k, where=where)

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


def count_tokens(text: str, model: str | None = None) -> int:
    model_name = model or settings.gemini_model
    response = client.models.count_tokens(
        model=model_name,
        contents=text,
    )
    total = getattr(response, "total_tokens", None)
    if total is None and isinstance(response, dict):
        total = response.get("total_tokens")
    return int(total or 0)


def get_sources(limit: int | None = None) -> list[str]:
    return list_sources(limit=limit)


def reset_db() -> dict[str, object]:
    return reset_store()


def _build_where(
    source: str | None,
    sources: list[str] | None,
    page_gte: int | None,
    page_lte: int | None,
) -> dict[str, object] | None:
    clauses: list[dict[str, object]] = []
    normalized_sources = [item for item in (sources or []) if item] if sources else []
    if not normalized_sources and source:
        normalized_sources = [source]
    if normalized_sources:
        clauses.append(
            {
                "source": normalized_sources[0]
                if len(normalized_sources) == 1
                else {"$in": normalized_sources}
            }
        )

    if page_gte is not None:
        clauses.append({"page": {"$gte": page_gte}})
    if page_lte is not None:
        clauses.append({"page": {"$lte": page_lte}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _chunked(items: list[DocumentIn], size: int) -> Iterable[list[DocumentIn]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _split_documents(
    documents: list[DocumentIn],
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[DocumentIn]:
    if chunk_tokens <= 0:
        return documents

    overlap = max(0, min(chunk_overlap, max(0, chunk_tokens - 1)))
    chunked: list[DocumentIn] = []
    for doc in documents:
        text = doc.text or ""
        if not text.strip():
            continue

        spans = _chunk_text(text, chunk_tokens, overlap)
        total = len(spans)
        if total == 0:
            continue

        parent_id = doc.id
        for index, span in enumerate(spans, start=1):
            raw_text = text[span.start : span.end]
            trimmed, trim_start, trim_end = _trim_span(raw_text, span.start, span.end)
            if not trimmed:
                continue
            metadata = _merge_chunk_metadata(
                doc.metadata,
                parent_id,
                index,
                total,
                trim_start,
                trim_end,
                span.tokens,
            )
            chunk_id: str | None = None
            if parent_id:
                chunk_id = parent_id if total == 1 else f"{parent_id}::chunk-{index}"
            chunked.append(
                DocumentIn(
                    id=chunk_id,
                    text=trimmed,
                    metadata=metadata,
                )
            )
    return chunked


def _chunk_text(
    text: str,
    chunk_tokens: int,
    chunk_overlap: int,
) -> list["_ChunkSpan"]:
    segments = _segments_from_text(text, chunk_tokens, chunk_overlap)
    if not segments:
        return []

    overlap = max(0, min(chunk_overlap, max(0, chunk_tokens - 1)))
    spans: list[_ChunkSpan] = []
    index = 0
    while index < len(segments):
        token_total = 0
        end_index = index
        while end_index < len(segments):
            segment_tokens = segments[end_index].tokens
            if token_total + segment_tokens > chunk_tokens and end_index > index:
                break
            token_total += segment_tokens
            end_index += 1

        last_index = end_index - 1
        span_start = segments[index].start
        span_end = segments[last_index].end
        spans.append(_ChunkSpan(span_start, span_end, token_total))

        if end_index >= len(segments):
            break

        if overlap <= 0:
            index = end_index
            continue

        back_tokens = 0
        overlap_index = last_index
        while overlap_index >= index:
            segment_tokens = segments[overlap_index].tokens
            if back_tokens + segment_tokens > overlap:
                break
            back_tokens += segment_tokens
            overlap_index -= 1

        next_index = max(overlap_index + 1, index + 1)
        if next_index <= index:
            next_index = index + 1
        index = next_index

    return spans


@dataclass(frozen=True)
class _Segment:
    start: int
    end: int
    tokens: int


@dataclass(frozen=True)
class _ChunkSpan:
    start: int
    end: int
    tokens: int


def _segments_from_text(
    text: str,
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[_Segment]:
    sentence_spans = _sentence_spans(text)
    if not sentence_spans:
        sentence_spans = [(0, len(text))]

    segments: list[_Segment] = []
    for start, end in sentence_spans:
        segment_text = text[start:end]
        token_count = _count_tokens(segment_text)
        if token_count <= 0:
            continue
        if token_count > chunk_tokens:
            segments.extend(
                _split_span_by_tokens(
                    text,
                    start,
                    end,
                    chunk_tokens,
                    chunk_overlap,
                )
            )
        else:
            segments.append(_Segment(start, end, token_count))
    return segments


def _split_span_by_tokens(
    text: str,
    span_start: int,
    span_end: int,
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[_Segment]:
    tokens = _token_spans(text[span_start:span_end])
    if not tokens:
        return []

    overlap = max(0, min(chunk_overlap, max(0, chunk_tokens - 1)))
    segments: list[_Segment] = []
    index = 0
    while index < len(tokens):
        end_index = min(index + chunk_tokens, len(tokens))
        start_token = tokens[index][0] + span_start
        end_token = tokens[end_index - 1][1] + span_start
        segments.append(_Segment(start_token, end_token, end_index - index))

        next_index = end_index - overlap if overlap > 0 else end_index
        if next_index <= index:
            next_index = index + 1
        index = next_index

    return segments


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE)]


def _count_tokens(text: str) -> int:
    return sum(1 for _ in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE))


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pattern = re.compile(r".*?(?:[.!?]+|\n+|$)", re.DOTALL)
    for match in pattern.finditer(text):
        start, end = match.span()
        segment = text[start:end]
        if not segment.strip():
            continue
        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment) - len(segment.rstrip())
        span_start = start + leading
        span_end = end - trailing
        if span_start < span_end:
            spans.append((span_start, span_end))
    return spans


def _trim_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    if not text:
        return "", start, end
    leading = len(text) - len(text.lstrip())
    trailing = len(text) - len(text.rstrip())
    trimmed = text.strip()
    return trimmed, start + leading, end - trailing


def _merge_chunk_metadata(
    metadata: dict[str, object] | None,
    parent_id: str | None,
    index: int,
    total: int,
    start: int,
    end: int,
    tokens: int,
) -> dict[str, object]:
    merged = dict(metadata) if isinstance(metadata, dict) else {}
    if parent_id:
        merged.setdefault("parent_id", parent_id)
    merged["chunk_index"] = index
    merged["chunk_total"] = total
    merged["chunk_start"] = start
    merged["chunk_end"] = end
    merged["chunk_tokens"] = tokens
    return merged


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
