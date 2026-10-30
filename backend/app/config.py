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

    # Office conversion (DOC, DOCX, PPT, PPTX, XLS) needs LibreOffice on the machine.
    libreoffice_bin: str = "soffice"

    # Vision: describes images (standalone uploads and figures in PDFs) so they can be retrieved.
    # Off by default. Image uploads are rejected without it, since nothing could be searched.
    enable_kg: bool = True          # builds the knowledge graph at upload; one LLM call per chunk
    kg_path: str = "./data/kg.json"
    kg_workers: int = 2             # concurrent extraction calls; keep low for free-tier rate limits
    enable_communities: bool = True  # GraphRAG: clusters the graph and summarizes each cluster (one LLM call each)

    # Parent-child chunking: small child chunks are searched, larger parents are returned as context
    parent_group_size: int = 3      # child chunks per parent, within one page

    # Agentic RAG
    agentic_max_steps: int = 3

    # Rerankers besides the default cross-encoder
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"

    enable_vlm: bool = False
    vision_model: str = "mistral-small-latest"  # must be vision-capable; verify against Mistral's model list
    vlm_workers: int = 2            # concurrent vision calls during ingestion
    vlm_query_max_images: int = 3   # figures sent to the vision model per question
    image_dir: str = "./data/images"  # stored figure bytes, used by VLM-enhanced queries

    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
