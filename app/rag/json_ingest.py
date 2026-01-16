from __future__ import annotations

import json
from typing import Any

from app.schemas import DocumentIn


def parse_json_documents(filename: str, json_bytes: bytes) -> list[DocumentIn]:
    payload = json_bytes.decode("utf-8-sig")
    data = json.loads(payload)
    return _build_documents(data, filename)


def _build_documents(data: Any, filename: str) -> list[DocumentIn]:
    documents: list[DocumentIn] = []

    if isinstance(data, dict) and "documents" in data:
        items = data.get("documents")
        if isinstance(items, list):
            for index, item in enumerate(items, start=1):
                doc = _normalize_item(item, filename, index)
                if doc:
                    documents.append(doc)
        return documents

    if isinstance(data, list):
        for index, item in enumerate(data, start=1):
            doc = _normalize_item(item, filename, index)
            if doc:
                documents.append(doc)
        return documents

    doc = _normalize_item(data, filename, 1)
    return [doc] if doc else []


def _normalize_item(item: Any, filename: str, index: int) -> DocumentIn | None:
    if item is None:
        return None

    if isinstance(item, str):
        return DocumentIn(text=item, metadata=_default_meta(filename, index))

    if isinstance(item, (int, float, bool)):
        return DocumentIn(text=str(item), metadata=_default_meta(filename, index))

    if isinstance(item, dict):
        if "text" in item:
            text = item.get("text")
            if text is None:
                return None
            metadata = item.get("metadata")
            metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
            if filename:
                metadata_dict.setdefault("source", filename)
            metadata_dict.setdefault("index", index)
            return DocumentIn(
                id=item.get("id"),
                text=str(text),
                metadata=metadata_dict,
            )

        return DocumentIn(
            text=json.dumps(item, ensure_ascii=False, separators=(",", ":")),
            metadata=_default_meta(filename, index),
        )

    if isinstance(item, (list, tuple)):
        return DocumentIn(
            text=json.dumps(item, ensure_ascii=False, separators=(",", ":")),
            metadata=_default_meta(filename, index),
        )

    return DocumentIn(text=str(item), metadata=_default_meta(filename, index))


def _default_meta(filename: str, index: int) -> dict[str, object]:
    meta: dict[str, object] = {"index": index}
    if filename:
        meta["source"] = filename
    return meta
