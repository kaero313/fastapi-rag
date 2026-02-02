from pathlib import Path
from typing import Any

import chromadb

from app.core.config import settings


def _ensure_persist_dir() -> None:
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


def get_collection():
    _ensure_persist_dir()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]] | None,
) -> None:
    collection = get_collection()
    if metadatas is None:
        collection.add(ids=ids, documents=texts, embeddings=embeddings)
    else:
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )


def query_by_embedding(
    embedding: list[float],
    top_k: int,
    where: dict[str, object] | None = None,
):
    collection = get_collection()
    params: dict[str, object] = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        params["where"] = where
    return collection.query(**params)


def list_sources(limit: int | None = None) -> list[str]:
    collection = get_collection()
    total = collection.count()
    if total <= 0:
        return []

    fetch_limit = total if limit is None else max(0, min(limit, total))
    if fetch_limit <= 0:
        return []

    data = collection.get(
        limit=fetch_limit,
        include=["metadatas"],
    )
    metadatas = data.get("metadatas") or []
    sources = {
        meta.get("source")
        for meta in metadatas
        if isinstance(meta, dict) and meta.get("source")
    }
    return sorted(sources)


def reset_store() -> dict[str, object]:
    _ensure_persist_dir()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    errors: list[str] = []

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    before_count = collection.count()

    reset_ok = False
    try:
        reset_ok = bool(client.reset())
    except Exception as exc:
        errors.append(str(exc))

    if not reset_ok:
        try:
            client.delete_collection(name=settings.chroma_collection)
            reset_ok = True
        except Exception as exc:
            errors.append(str(exc))

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    after_count = collection.count()

    result: dict[str, object] = {
        "collection": collection.name,
        "before_count": before_count,
        "after_count": after_count,
        "reset": reset_ok,
    }
    if errors:
        result["errors"] = errors
    return result
