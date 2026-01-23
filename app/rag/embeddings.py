import time

from google import genai
from google.genai import types

from app.core.config import settings

client = genai.Client(api_key=settings.gemini_api_key)


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
            response = client.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings = response.embeddings or []
            if not embeddings or embeddings[0].values is None:
                raise ValueError("Empty embedding response.")
            return embeddings[0].values
        except Exception:
            if attempt + 1 >= max_retries:
                raise
            time.sleep(backoff * (2 ** attempt))
