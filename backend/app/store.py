from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import get_settings
from .retrieval import Index

_index: Index | None = None


def get_store() -> Index:
    """Load models and the persisted index once per process."""
    global _index
    if _index is None:
        s = get_settings()
        embedder = SentenceTransformer(s.embed_model)
        reranker = CrossEncoder(s.rerank_model)
        _index = Index(
            embedder,
            QdrantClient(path=s.qdrant_path),
            s.collection_name,
            embedder.get_sentence_embedding_dimension(),
            reranker,
        )
    return _index
