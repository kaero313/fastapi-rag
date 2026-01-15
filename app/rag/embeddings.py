import google.generativeai as genai

from app.core.config import settings

genai.configure(api_key=settings.gemini_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    embeddings: list[list[float]] = []
    for text in texts:
        response = genai.embed_content(
            model=settings.gemini_embedding_model,
            content=text,
            task_type="retrieval_document",
        )
        embeddings.append(response["embedding"])
    return embeddings
