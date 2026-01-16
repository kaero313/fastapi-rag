from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app.rag.json_ingest import parse_json_documents
from app.rag.pdf import extract_text_by_page
from app.schemas import DocumentIn

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".json",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".log",
}


def normalize_extensions(extensions: list[str] | None) -> set[str] | None:
    if not extensions:
        return None
    normalized: set[str] = set()
    for ext in extensions:
        value = ext.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = "." + value
        normalized.add(value)
    return normalized or None


def load_documents_from_directory(
    directory: Path,
    recursive: bool = True,
    extensions: list[str] | None = None,
    base_dir: Path | None = None,
) -> tuple[list[DocumentIn], dict[str, object]]:
    allowed_exts = normalize_extensions(extensions)
    documents: list[DocumentIn] = []
    files_processed = 0
    files_skipped = 0
    files_failed = 0
    files_empty = 0
    error_files: list[str] = []

    files = _iter_files(directory, recursive)
    for path in files:
        ext = path.suffix.lower()
        if allowed_exts is None and ext not in SUPPORTED_EXTENSIONS:
            files_skipped += 1
            continue
        if allowed_exts is not None and ext not in allowed_exts:
            files_skipped += 1
            continue

        files_processed += 1
        try:
            source_label = _source_label(path, base_dir)
            docs = _load_documents_from_file(path, ext, source_label)
            if not docs:
                files_empty += 1
                continue
            documents.extend(docs)
        except Exception:
            files_failed += 1
            if len(error_files) < 5:
                error_files.append(str(path))

    stats: dict[str, object] = {
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "files_empty": files_empty,
        "files_failed": files_failed,
    }
    if error_files:
        stats["error_files"] = error_files
    return documents, stats


def _iter_files(directory: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in directory.glob(pattern):
        if path.is_file():
            yield path


def _load_documents_from_file(
    path: Path,
    ext: str,
    source_label: str,
) -> list[DocumentIn]:
    if ext == ".pdf":
        return _documents_from_pdf(path, source_label)
    if ext == ".json":
        return _documents_from_json(path, source_label)
    if ext in {".txt", ".md", ".markdown", ".csv", ".log"}:
        return _documents_from_text(path, source_label)
    return []


def _documents_from_pdf(path: Path, source_label: str) -> list[DocumentIn]:
    pdf_bytes = path.read_bytes()
    pages = extract_text_by_page(pdf_bytes)
    documents: list[DocumentIn] = []
    for index, text in enumerate(pages, start=1):
        documents.append(
            DocumentIn(
                text=text,
                metadata={"source": source_label, "page": index},
            )
        )
    return documents


def _documents_from_json(path: Path, source_label: str) -> list[DocumentIn]:
    json_bytes = path.read_bytes()
    try:
        return parse_json_documents(source_label, json_bytes)
    except json.JSONDecodeError:
        return []


def _documents_from_text(path: Path, source_label: str) -> list[DocumentIn]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [DocumentIn(text=text, metadata={"source": source_label})]


def _source_label(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return path.name
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return path.name
