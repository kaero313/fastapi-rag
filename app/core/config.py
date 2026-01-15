from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "text-embedding-004"
    chroma_persist_dir: str = "data/chroma"
    chroma_collection: str = "rag"
    top_k: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
