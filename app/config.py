from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"  # "groq" or "mistral"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    qdrant_path: str = "./data/qdrant"
    collection_name: str = "documents"

    chunk_size: int = 800
    chunk_overlap: int = 120
    dense_k: int = 20          # candidates from vector search
    bm25_k: int = 20           # candidates from keyword search
    rerank_top_n: int = 5      # passages sent to the LLM
    rerank_min_score: float = -3.0  # below this, refuse instead of answering
    max_upload_mb: int = 20

    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
