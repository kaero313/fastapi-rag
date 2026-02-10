# __future__에서 항목을 가져옵니다.
from __future__ import annotations

# dataclasses에서 항목을 가져옵니다.
from dataclasses import dataclass
# pathlib에서 항목을 가져옵니다.
from pathlib import Path
# re 모듈을 가져옵니다.
import re
# typing에서 항목을 가져옵니다.
from typing import Iterable
# uuid에서 항목을 가져옵니다.
from uuid import uuid4

# google에서 항목을 가져옵니다.
from google import genai
# google.genai에서 항목을 가져옵니다.
from google.genai import types

# app.core.config에서 항목을 가져옵니다.
from app.core.config import settings
# app.rag.directory_ingest에서 항목을 가져옵니다.
from app.rag.directory_ingest import load_documents_from_directory
# app.rag.embeddings에서 항목을 가져옵니다.
from app.rag.embeddings import embed_texts
# app.rag.json_ingest에서 항목을 가져옵니다.
from app.rag.json_ingest import parse_json_documents
# app.rag.pdf에서 항목을 가져옵니다.
from app.rag.pdf import extract_text_by_page
# app.rag.vectorstore에서 항목을 가져옵니다.
from app.rag.vectorstore import (
    # 다음 인자를 전달합니다.
    add_documents,
    # 다음 인자를 전달합니다.
    list_sources,
    # 다음 인자를 전달합니다.
    query_by_embedding,
    # 다음 인자를 전달합니다.
    reset_store,
# 여러 줄 구문을 닫습니다.
)
# app.schemas에서 항목을 가져옵니다.
from app.schemas import DocumentIn, QueryResponse, Source

# SYSTEM_PROMPT에 대입하는 호출을 시작합니다.
SYSTEM_PROMPT = (
    # 코드를 실행합니다.
    "You are a helpful assistant. Use the provided context to answer the question. "
    # 코드를 실행합니다.
    "If the answer is not in the context, say you do not know."
# 여러 줄 구문을 닫습니다.
)

# client에 값을 대입합니다.
client = genai.Client(api_key=settings.gemini_api_key)


# ingest_documents 함수를 정의합니다.
def ingest_documents(documents: list[DocumentIn]) -> list[str]:
    # ids: list[str]에 값을 대입합니다.
    ids: list[str] = []
    # chunk_tokens에 값을 대입합니다.
    chunk_tokens = max(0, settings.ingest_chunk_size)
    # chunk_overlap에 값을 대입합니다.
    chunk_overlap = max(0, settings.ingest_chunk_overlap)
    # expanded_documents에 값을 대입합니다.
    expanded_documents = _split_documents(documents, chunk_tokens, chunk_overlap)
    # include_metadata에 값을 대입합니다.
    include_metadata = any(doc.metadata for doc in expanded_documents)

    # batch_size에 값을 대입합니다.
    batch_size = max(1, settings.ingest_batch_size)
    # 반복문을 시작합니다.
    for batch in _chunked(expanded_documents, batch_size):
        # texts: list[str]에 값을 대입합니다.
        texts: list[str] = []
        # metadatas: list[dict[str, object]]에 값을 대입합니다.
        metadatas: list[dict[str, object]] = []
        # batch_ids: list[str]에 값을 대입합니다.
        batch_ids: list[str] = []

        # 반복문을 시작합니다.
        for doc in batch:
            # doc_id에 값을 대입합니다.
            doc_id = doc.id or uuid4().hex
            # 리스트에 값을 추가합니다.
            batch_ids.append(doc_id)
            # 리스트에 값을 추가합니다.
            texts.append(doc.text)
            # 조건을 검사합니다.
            if include_metadata:
                # metadata에 값을 대입합니다.
                metadata = doc.metadata or {"_missing": True}
                # 리스트에 값을 추가합니다.
                metadatas.append(metadata)

        # embeddings에 값을 대입합니다.
        embeddings = embed_texts(texts, task_type="retrieval_document")
        # 여러 줄 호출을 시작합니다.
        add_documents(
            # ids 인자를 전달합니다.
            ids=batch_ids,
            # texts 인자를 전달합니다.
            texts=texts,
            # embeddings 인자를 전달합니다.
            embeddings=embeddings,
            # metadatas 인자를 전달합니다.
            metadatas=metadatas if include_metadata else None,
        # 여러 줄 구문을 닫습니다.
        )
        # 리스트에 여러 값을 추가합니다.
        ids.extend(batch_ids)
    # 값을 반환합니다.
    return ids


# ingest_pdf_bytes 함수를 정의합니다.
def ingest_pdf_bytes(filename: str, pdf_bytes: bytes) -> list[str]:
    # pages에 값을 대입합니다.
    pages = extract_text_by_page(pdf_bytes)
    # documents: list[DocumentIn]에 값을 대입합니다.
    documents: list[DocumentIn] = []
    # 반복문을 시작합니다.
    for index, text in enumerate(pages, start=1):
        # 여러 줄 호출을 시작합니다.
        documents.append(
            # 여러 줄 호출을 시작합니다.
            DocumentIn(
                # text 인자를 전달합니다.
                text=text,
                # metadata 인자를 전달합니다.
                metadata={"source": filename, "page": index},
            # 여러 줄 구문을 닫습니다.
            )
        # 여러 줄 구문을 닫습니다.
        )
    # 조건을 검사합니다.
    if not documents:
        # 값을 반환합니다.
        return []
    # 값을 반환합니다.
    return ingest_documents(documents)


# ingest_json_bytes 함수를 정의합니다.
def ingest_json_bytes(filename: str, json_bytes: bytes) -> list[str]:
    # documents에 값을 대입합니다.
    documents = parse_json_documents(filename, json_bytes)
    # 조건을 검사합니다.
    if not documents:
        # 값을 반환합니다.
        return []
    # 값을 반환합니다.
    return ingest_documents(documents)


# ingest_directory 함수를 정의합니다.
def ingest_directory(
    # directory 매개변수입니다.
    directory: Path,
    # recursive: bool 인자를 전달합니다.
    recursive: bool = True,
    # extensions: list[str] | None 인자를 전달합니다.
    extensions: list[str] | None = None,
    # base_dir: Path | None 인자를 전달합니다.
    base_dir: Path | None = None,
# 함수 시그니처를 닫고 반환 타입을 적습니다.
) -> tuple[list[str], dict[str, object]]:
    # documents, stats에 대입하는 호출을 시작합니다.
    documents, stats = load_documents_from_directory(
        # 다음 인자를 전달합니다.
        directory,
        # recursive 인자를 전달합니다.
        recursive=recursive,
        # extensions 인자를 전달합니다.
        extensions=extensions,
        # base_dir 인자를 전달합니다.
        base_dir=base_dir,
    # 여러 줄 구문을 닫습니다.
    )
    # 조건을 검사합니다.
    if not documents:
        # 값을 반환합니다.
        return [], stats
    # ids에 값을 대입합니다.
    ids = ingest_documents(documents)
    # 값을 반환합니다.
    return ids, stats


# answer_query 함수를 정의합니다.
def answer_query(
    # query 매개변수입니다.
    query: str,
    # top_k: int | None 인자를 전달합니다.
    top_k: int | None = None,
    # source: str | None 인자를 전달합니다.
    source: str | None = None,
    # sources: list[str] | None 인자를 전달합니다.
    sources: list[str] | None = None,
    # page_gte: int | None 인자를 전달합니다.
    page_gte: int | None = None,
    # page_lte: int | None 인자를 전달합니다.
    page_lte: int | None = None,
# 함수 시그니처를 닫고 반환 타입을 적습니다.
) -> QueryResponse:
    # effective_top_k에 값을 대입합니다.
    effective_top_k = top_k or settings.top_k
    # query_embedding에 값을 대입합니다.
    query_embedding = embed_texts([query], task_type="retrieval_query")[0]
    # multiplier에 값을 대입합니다.
    multiplier = max(1, settings.candidate_k_multiplier)
    # candidate_k에 대입하는 호출을 시작합니다.
    candidate_k = max(
        # 다음 인자를 전달합니다.
        effective_top_k * multiplier,
        # 다음 인자를 전달합니다.
        settings.candidate_k_min,
    # 여러 줄 구문을 닫습니다.
    )
    # where에 값을 대입합니다.
    where = _build_where(source, sources, page_gte, page_lte)
    # results에 값을 대입합니다.
    results = query_by_embedding(query_embedding, top_k=candidate_k, where=where)

    # ids에 값을 대입합니다.
    ids = results.get("ids", [[]])[0]
    # documents에 값을 대입합니다.
    documents = results.get("documents", [[]])[0]
    # metadatas에 값을 대입합니다.
    metadatas = results.get("metadatas", [[]])[0]
    # distances에 값을 대입합니다.
    distances = results.get("distances", [[]])[0]

    # sources: list[Source]에 값을 대입합니다.
    sources: list[Source] = []
    # context_chunks: list[str]에 값을 대입합니다.
    context_chunks: list[str] = []

    # ranked에 값을 대입합니다.
    ranked = _rerank_results(query, ids, documents, metadatas, distances)
    # 반복문을 시작합니다.
    for doc_id, text, metadata, distance in ranked[:effective_top_k]:
        # score에 값을 대입합니다.
        score = 1.0 - float(distance) if distance is not None else 0.0
        # 여러 줄 호출을 시작합니다.
        sources.append(
            # Source(id에 값을 대입합니다.
            Source(id=doc_id, score=score, text=text, metadata=metadata)
        # 여러 줄 구문을 닫습니다.
        )
        # 리스트에 값을 추가합니다.
        context_chunks.append(text)

    # context에 값을 대입합니다.
    context = "\n\n".join(context_chunks)

    # prompt에 값을 대입합니다.
    prompt = "Context:\n" + context + "\n\n" + "Question:\n" + query
    # response에 대입하는 호출을 시작합니다.
    response = client.models.generate_content(
        # model 인자를 전달합니다.
        model=settings.gemini_model,
        # contents 인자를 전달합니다.
        contents=prompt,
        # config 인자를 전달합니다.
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    # 여러 줄 구문을 닫습니다.
    )
    # answer에 값을 대입합니다.
    answer = response.text or ""
    # 값을 반환합니다.
    return QueryResponse(answer=answer, sources=sources)


# count_tokens 함수를 정의합니다.
def count_tokens(text: str, model: str | None = None) -> int:
    # model_name에 값을 대입합니다.
    model_name = model or settings.gemini_model
    # response에 대입하는 호출을 시작합니다.
    response = client.models.count_tokens(
        # model 인자를 전달합니다.
        model=model_name,
        # contents 인자를 전달합니다.
        contents=text,
    # 여러 줄 구문을 닫습니다.
    )
    # total에 값을 대입합니다.
    total = getattr(response, "total_tokens", None)
    # 조건을 검사합니다.
    if total is None and isinstance(response, dict):
        # total에 값을 대입합니다.
        total = response.get("total_tokens")
    # 값을 반환합니다.
    return int(total or 0)


# get_sources 함수를 정의합니다.
def get_sources(limit: int | None = None) -> list[str]:
    # 값을 반환합니다.
    return list_sources(limit=limit)


# reset_db 함수를 정의합니다.
def reset_db() -> dict[str, object]:
    # 값을 반환합니다.
    return reset_store()


# _build_where 함수를 정의합니다.
def _build_where(
    # source 매개변수입니다.
    source: str | None,
    # sources 매개변수입니다.
    sources: list[str] | None,
    # page_gte 매개변수입니다.
    page_gte: int | None,
    # page_lte 매개변수입니다.
    page_lte: int | None,
# 함수 시그니처를 닫고 반환 타입을 적습니다.
) -> dict[str, object] | None:
    # clauses: list[dict[str, object]]에 값을 대입합니다.
    clauses: list[dict[str, object]] = []
    # normalized_sources에 값을 대입합니다.
    normalized_sources = [item for item in (sources or []) if item] if sources else []
    # 조건을 검사합니다.
    if not normalized_sources and source:
        # normalized_sources에 값을 대입합니다.
        normalized_sources = [source]
    # 조건을 검사합니다.
    if normalized_sources:
        # 여러 줄 호출을 시작합니다.
        clauses.append(
            # 딕셔너리 리터럴을 시작합니다.
            {
                # 코드를 실행합니다.
                "source": normalized_sources[0]
                # 조건을 검사합니다.
                if len(normalized_sources) == 1
                # 코드를 실행합니다.
                else {"$in": normalized_sources}
            # 여러 줄 구문을 닫습니다.
            }
        # 여러 줄 구문을 닫습니다.
        )

    # 조건을 검사합니다.
    if page_gte is not None:
        # 리스트에 값을 추가합니다.
        clauses.append({"page": {"$gte": page_gte}})
    # 조건을 검사합니다.
    if page_lte is not None:
        # 리스트에 값을 추가합니다.
        clauses.append({"page": {"$lte": page_lte}})

    # 조건을 검사합니다.
    if not clauses:
        # 값을 반환합니다.
        return None
    # 조건을 검사합니다.
    if len(clauses) == 1:
        # 값을 반환합니다.
        return clauses[0]
    # 값을 반환합니다.
    return {"$and": clauses}


# _chunked 함수를 정의합니다.
def _chunked(items: list[DocumentIn], size: int) -> Iterable[list[DocumentIn]]:
    # 반복문을 시작합니다.
    for index in range(0, len(items), size):
        # 제너레이터에서 값을 반환합니다.
        yield items[index : index + size]


# _split_documents 함수를 정의합니다.
def _split_documents(
    # documents 매개변수입니다.
    documents: list[DocumentIn],
    # chunk_tokens 매개변수입니다.
    chunk_tokens: int,
    # chunk_overlap 매개변수입니다.
    chunk_overlap: int,
# 함수 시그니처를 닫고 반환 타입을 적습니다.
) -> list[DocumentIn]:
    # 조건을 검사합니다.
    if chunk_tokens <= 0:
        # 값을 반환합니다.
        return documents

    # overlap에 값을 대입합니다.
    overlap = max(0, min(chunk_overlap, max(0, chunk_tokens - 1)))
    # chunked: list[DocumentIn]에 값을 대입합니다.
    chunked: list[DocumentIn] = []
    # 반복문을 시작합니다.
    for doc in documents:
        # text에 값을 대입합니다.
        text = doc.text or ""
        # 조건을 검사합니다.
        if not text.strip():
            # 이번 반복을 건너뜁니다.
            continue

        # spans에 값을 대입합니다.
        spans = _chunk_text(text, chunk_tokens, overlap)
        # total에 값을 대입합니다.
        total = len(spans)
        # 조건을 검사합니다.
        if total == 0:
            # 이번 반복을 건너뜁니다.
            continue

        # parent_id에 값을 대입합니다.
        parent_id = doc.id
        # 반복문을 시작합니다.
        for index, span in enumerate(spans, start=1):
            # raw_text에 값을 대입합니다.
            raw_text = text[span.start : span.end]
            # trimmed, trim_start, trim_end에 값을 대입합니다.
            trimmed, trim_start, trim_end = _trim_span(raw_text, span.start, span.end)
            # 조건을 검사합니다.
            if not trimmed:
                # Skip to the next iteration.
                continue
            # Start a multi-line call assigned to metadata.
            metadata = _merge_chunk_metadata(
                # Pass the next argument.
                doc.metadata,
                # Pass the next argument.
                parent_id,
                # Pass the next argument.
                index,
                # Pass the next argument.
                total,
                # Pass the next argument.
                trim_start,
                # Pass the next argument.
                trim_end,
                # Pass the next argument.
                span.tokens,
            # End a multi-line block.
            )
            # Set chunk_id: str | None.
            chunk_id: str | None = None
            # Check a condition.
            if parent_id:
                # Execute this statement.
                chunk_id = parent_id if total == 1 else f"{parent_id}::chunk-{index}"
            # Start a multi-line call.
            chunked.append(
                # Start a multi-line call.
                DocumentIn(
                    # Pass the id argument.
                    id=chunk_id,
                    # Pass the text argument.
                    text=trimmed,
                    # Pass the metadata argument.
                    metadata=metadata,
                # End a multi-line block.
                )
            # End a multi-line block.
            )
    # Return a result.
    return chunked


# Define the _chunk_text function.
def _chunk_text(
    # Function parameter text.
    text: str,
    # Function parameter chunk_tokens.
    chunk_tokens: int,
    # Function parameter chunk_overlap.
    chunk_overlap: int,
# Close the signature and declare the return type.
) -> list["_ChunkSpan"]:
    # Set segments.
    segments = _segments_from_text(text, chunk_tokens, chunk_overlap)
    # Check a condition.
    if not segments:
        # Return a result.
        return []

    # Set overlap.
    overlap = max(0, min(chunk_overlap, max(0, chunk_tokens - 1)))
    # Set spans: list[_ChunkSpan].
    spans: list[_ChunkSpan] = []
    # Set index.
    index = 0
    # Loop while the condition is true.
    while index < len(segments):
        # Set token_total.
        token_total = 0
        # Set end_index.
        end_index = index
        # Loop while the condition is true.
        while end_index < len(segments):
            # Set segment_tokens.
            segment_tokens = segments[end_index].tokens
            # Check a condition.
            if token_total + segment_tokens > chunk_tokens and end_index > index:
                # Exit the loop.
                break
            # Set token_total +.
            token_total += segment_tokens
            # Set end_index +.
            end_index += 1

        # Set last_index.
        last_index = end_index - 1
        # Set span_start.
        span_start = segments[index].start
        # Set span_end.
        span_end = segments[last_index].end
        # Append an item to a list.
        spans.append(_ChunkSpan(span_start, span_end, token_total))

        # Check a condition.
        if end_index >= len(segments):
            # Exit the loop.
            break

        # Check a condition.
        if overlap <= 0:
            # Set index.
            index = end_index
            # Skip to the next iteration.
            continue

        # Set back_tokens.
        back_tokens = 0
        # Set overlap_index.
        overlap_index = last_index
        # Loop while the condition is true.
        while overlap_index >= index:
            # Set segment_tokens.
            segment_tokens = segments[overlap_index].tokens
            # Check a condition.
            if back_tokens + segment_tokens > overlap:
                # Exit the loop.
                break
            # Set back_tokens +.
            back_tokens += segment_tokens
            # Set overlap_index -.
            overlap_index -= 1

        # Set next_index.
        next_index = max(overlap_index + 1, index + 1)
        # Check a condition.
        if next_index <= index:
            # Set next_index.
            next_index = index + 1
        # Set index.
        index = next_index

    # Return a result.
    return spans


# Apply a decorator.
@dataclass(frozen=True)
class _Segment:  # Define the _Segment class.
    # Declare field start.
    start: int
    # Declare field end.
    end: int
    # Declare field tokens.
    tokens: int


# Apply a decorator.
@dataclass(frozen=True)
class _ChunkSpan:  # Define the _ChunkSpan class.
    # Declare field start.
    start: int
    # Declare field end.
    end: int
    # Declare field tokens.
    tokens: int


# Define the _segments_from_text function.
def _segments_from_text(
    # Function parameter text.
    text: str,
    # Function parameter chunk_tokens.
    chunk_tokens: int,
    # Function parameter chunk_overlap.
    chunk_overlap: int,
# Close the signature and declare the return type.
) -> list[_Segment]:
    # Set sentence_spans.
    sentence_spans = _sentence_spans(text)
    # Check a condition.
    if not sentence_spans:
        # Set sentence_spans.
        sentence_spans = [(0, len(text))]

    # Set segments: list[_Segment].
    segments: list[_Segment] = []
    # Start a loop.
    for start, end in sentence_spans:
        # Set segment_text.
        segment_text = text[start:end]
        # Set token_count.
        token_count = _count_tokens(segment_text)
        # Check a condition.
        if token_count <= 0:
            # Skip to the next iteration.
            continue
        # Check a condition.
        if token_count > chunk_tokens:
            # Start a multi-line call.
            segments.extend(
                # Start a multi-line call.
                _split_span_by_tokens(
                    # Pass the next argument.
                    text,
                    # Pass the next argument.
                    start,
                    # Pass the next argument.
                    end,
                    # Pass the next argument.
                    chunk_tokens,
                    # Pass the next argument.
                    chunk_overlap,
                # End a multi-line block.
                )
            # End a multi-line block.
            )
        # Fallback branch.
        else:
            # Append an item to a list.
            segments.append(_Segment(start, end, token_count))
    # Return a result.
    return segments


# Define the _split_span_by_tokens function.
def _split_span_by_tokens(
    # Function parameter text.
    text: str,
    # Function parameter span_start.
    span_start: int,
    # Function parameter span_end.
    span_end: int,
    # Function parameter chunk_tokens.
    chunk_tokens: int,
    # Function parameter chunk_overlap.
    chunk_overlap: int,
# Close the signature and declare the return type.
) -> list[_Segment]:
    # Set tokens.
    tokens = _token_spans(text[span_start:span_end])
    # Check a condition.
    if not tokens:
        # Return a result.
        return []

    # Set overlap.
    overlap = max(0, min(chunk_overlap, max(0, chunk_tokens - 1)))
    # Set segments: list[_Segment].
    segments: list[_Segment] = []
    # Set index.
    index = 0
    # Loop while the condition is true.
    while index < len(tokens):
        # Set end_index.
        end_index = min(index + chunk_tokens, len(tokens))
        # Set start_token.
        start_token = tokens[index][0] + span_start
        # Set end_token.
        end_token = tokens[end_index - 1][1] + span_start
        # Append an item to a list.
        segments.append(_Segment(start_token, end_token, end_index - index))

        # Set next_index.
        next_index = end_index - overlap if overlap > 0 else end_index
        # Check a condition.
        if next_index <= index:
            # Set next_index.
            next_index = index + 1
        # Set index.
        index = next_index

    # Return a result.
    return segments


# Define the _token_spans function.
def _token_spans(text: str) -> list[tuple[int, int]]:
    # Return a result.
    return [match.span() for match in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE)]


# Define the _count_tokens function.
def _count_tokens(text: str) -> int:
    # Return a result.
    return sum(1 for _ in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE))


# Define the _sentence_spans function.
def _sentence_spans(text: str) -> list[tuple[int, int]]:
    # Set spans: list[tuple[int, int]].
    spans: list[tuple[int, int]] = []
    # Set pattern.
    pattern = re.compile(r".*?(?:[.!?]+|\n+|$)", re.DOTALL)
    # Start a loop.
    for match in pattern.finditer(text):
        # Set start, end.
        start, end = match.span()
        # Set segment.
        segment = text[start:end]
        # Check a condition.
        if not segment.strip():
            # Skip to the next iteration.
            continue
        # Set leading.
        leading = len(segment) - len(segment.lstrip())
        # Set trailing.
        trailing = len(segment) - len(segment.rstrip())
        # Set span_start.
        span_start = start + leading
        # Set span_end.
        span_end = end - trailing
        # Check a condition.
        if span_start < span_end:
            # Append an item to a list.
            spans.append((span_start, span_end))
    # Return a result.
    return spans


# Define the _trim_span function.
def _trim_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    # Check a condition.
    if not text:
        # Return a result.
        return "", start, end
    # Set leading.
    leading = len(text) - len(text.lstrip())
    # Set trailing.
    trailing = len(text) - len(text.rstrip())
    # Set trimmed.
    trimmed = text.strip()
    # Return a result.
    return trimmed, start + leading, end - trailing


# Define the _merge_chunk_metadata function.
def _merge_chunk_metadata(
    # Function parameter metadata.
    metadata: dict[str, object] | None,
    # Function parameter parent_id.
    parent_id: str | None,
    # Function parameter index.
    index: int,
    # Function parameter total.
    total: int,
    # Function parameter start.
    start: int,
    # Function parameter end.
    end: int,
    # Function parameter tokens.
    tokens: int,
# Close the signature and declare the return type.
) -> dict[str, object]:
    # Set merged.
    merged = dict(metadata) if isinstance(metadata, dict) else {}
    # Check a condition.
    if parent_id:
        # Execute this statement.
        merged.setdefault("parent_id", parent_id)
    # Set merged["chunk_index"].
    merged["chunk_index"] = index
    # Set merged["chunk_total"].
    merged["chunk_total"] = total
    # Set merged["chunk_start"].
    merged["chunk_start"] = start
    # Set merged["chunk_end"].
    merged["chunk_end"] = end
    # Set merged["chunk_tokens"].
    merged["chunk_tokens"] = tokens
    # Return a result.
    return merged


# Define the _rerank_results function.
def _rerank_results(
    # Function parameter query.
    query: str,
    # Function parameter ids.
    ids: list[str],
    # Function parameter documents.
    documents: list[str],
    # Function parameter metadatas.
    metadatas: list[dict[str, object] | None],
    # Function parameter distances.
    distances: list[float | None],
# Close the signature and declare the return type.
) -> list[tuple[str, str, dict[str, object] | None, float | None]]:
    # Set query_lower.
    query_lower = query.lower()
    # Set terms.
    terms = [term for term in query_lower.split() if term]

    # Define the lexical_score function.
    def lexical_score(text: str) -> int:
        # Check a condition.
        if not text:
            # Return a result.
            return 0
        # Set lowered.
        lowered = text.lower()
        # Set score.
        score = 0
        # Check a condition.
        if query_lower and query_lower in lowered:
            # Set score +.
            score += len(terms) + 1
        # Start a loop.
        for term in terms:
            # Check a condition.
            if term in lowered:
                # Set score +.
                score += 1
        # Return a result.
        return score

    # Set scored: list[tuple[int, float, int, str, str, dict[str, object] | None, float | None]].
    scored: list[tuple[int, float, int, str, str, dict[str, object] | None, float | None]] = []
    # Start a multi-line call.
    for index, (doc_id, text, metadata, distance) in enumerate(
        # Execute this statement.
        zip(ids, documents, metadatas, distances)
    # Close the multi-line statement.
    ):
        # Set score.
        score = lexical_score(text)
        # Set dist_value.
        dist_value = float(distance) if distance is not None else 1.0
        # Append an item to a list.
        scored.append((score, dist_value, index, doc_id, text, metadata, distance))

    # Set scored.sort(key.
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    # Return a result.
    return [(doc_id, text, metadata, distance) for _, _, _, doc_id, text, metadata, distance in scored]
