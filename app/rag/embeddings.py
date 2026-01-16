import time

import google.generativeai as genai

from app.core.config import settings

genai.configure(api_key=settings.gemini_api_key)


def embed_texts(
    texts: list[str],
    task_type: str = "retrieval_document",
) -> list[list[float]]:
    if not texts:
        return []
    embeddings: list[list[float]] = []
    for text in texts:
        embeddings.append(_embed_with_retry(text, task_type))
    return embeddings


def _embed_with_retry(text: str, task_type: str) -> list[float]:
    max_retries = max(1, settings.embed_max_retries)
    backoff = max(0.0, settings.embed_retry_backoff)
    for attempt in range(max_retries):
        try:
            response = genai.embed_content(
                model=settings.gemini_embedding_model,
                content=text,
                task_type=task_type,
            )
            return response["embedding"]
        except Exception:
            if attempt + 1 >= max_retries:
                raise
            time.sleep(backoff * (2 ** attempt))
