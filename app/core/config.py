from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "text-embedding-004"
    embed_max_retries: int = 3
    embed_retry_backoff: float = 1.0
    ingest_batch_size: int = 64
    ingest_chunk_size: int = 4000
    ingest_chunk_overlap: int = 400
    candidate_k_multiplier: int = 10
    candidate_k_min: int = 100
    chroma_persist_dir: str = "data/chroma"
    chroma_collection: str = "rag"
    top_k: int = 4
    ingest_base_dir: str = "data/ingest"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
