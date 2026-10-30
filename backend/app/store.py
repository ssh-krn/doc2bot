from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import get_settings
from .kg.graph import KnowledgeGraph
from .retrieval import Index

_index: Index | None = None
_graph: KnowledgeGraph | None = None


def get_store() -> Index:
    """Load models and the persisted chunk index once per process."""
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


def get_graph() -> KnowledgeGraph | None:
    """The knowledge graph, or None when ENABLE_KG is off. Shares the index's embedder."""
    global _graph
    s = get_settings()
    if not s.enable_kg:
        return None
    if _graph is None:
        _graph = KnowledgeGraph(get_store().embedder, Path(s.kg_path))
    return _graph
