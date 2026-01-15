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


def query_by_embedding(embedding: list[float], top_k: int):
    collection = get_collection()
    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
