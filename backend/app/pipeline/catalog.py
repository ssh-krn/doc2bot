"""Names and descriptions shown in the UI. The hybrid labels state the RAG type and the search types used,
so the same technique reads differently under each mode."""

from .options import LIGHTRAG_MODES, MODES, Options

MODE_INFO = {
    "traditional": ("Traditional RAG", "Vector and keyword search over chunks, reranked, then one answer call."),
    "crag": ("CRAG (Corrective RAG)", "Grades each retrieved passage. If none are relevant, rewrites the query and retries. Refuses if still no evidence. No web-search fallback."),
    "self_rag": ("Self-RAG", "Keeps only passages judged relevant, then checks the drafted answer against its sources and regenerates it if unsupported. Retrieval-on-demand is not included."),
    "lightrag": ("LightRAG", "Graph retrieval over entities and relations (local, global or hybrid sub-mode), plus chunk search."),
    "agentic": ("Agentic RAG", "The model decides whether to search again, up to a fixed number of rounds, then answers from everything gathered."),
    "graphrag": ("GraphRAG", "Clusters the knowledge graph into communities, summarizes each one, and answers broad questions from those summaries plus their source chunks."),
}

LIGHTRAG_INFO = {
    "local": "Entity-centred: entities matching the question, with their relations and source chunks.",
    "global": "Relation-centred: relations matching the question, with their source chunks.",
    "hybrid": "Entities and relations together, with chunk search.",
}

TECHNIQUE_INFO = {
    "hybrid": ("Hybrid search", "Combines BM25 keyword search with dense vector search, merged by reciprocal rank fusion."),
    "query_rewriting": ("Query rewriting", "The model writes alternative phrasings of the question. Results for each phrasing are fused."),
    "contextual_compression": ("Contextual compression", "Keeps only the two most relevant sentences of each passage, scored by the reranker."),
    "parent_document": ("Parent document retriever", "Searches small child chunks, then answers with their larger parent passage."),
    "sentence_window": ("Sentence window retriever", "Finds the most relevant sentence in each passage and adds two sentences of context either side."),
    "auto_merging": ("Auto-merging retriever", "When two or more child chunks of one parent are retrieved, returns the parent instead."),
}

RERANKER_INFO = {
    "cross_encoder": ("Cross-encoder (default)", "Local model that reads each question-passage pair together."),
    "flashrank": ("FlashRank", "Lightweight local reranker. Needs the flashrank package."),
    "cohere": ("Cohere Rerank", "Hosted reranker. Needs COHERE_API_KEY, and sends passages to Cohere."),
    "bi_encoder": ("Bi-encoder similarity", "Cosine similarity of the question and passage embeddings. Fast, coarser than a cross-encoder."),
}

_HYBRID_SEARCH = {"traditional": "BM25 keyword + dense vector", "crag": "BM25 keyword + dense vector",
                  "self_rag": "BM25 keyword + dense vector", "agentic": "BM25 keyword + dense vector"}


def hybrid_label(opts: Options) -> str:
    """Name for the search used under this mode. Hybrid search is named after the RAG type and search types."""
    name = MODE_INFO[opts.mode][0]
    if opts.mode == "lightrag":
        search = f"entity and relation vectors ({opts.lightrag_mode}) + BM25 keyword + dense chunks"
    elif opts.mode == "graphrag":
        search = "community summary vectors + BM25 keyword + dense chunks"
    else:
        search = _HYBRID_SEARCH[opts.mode]
    if "hybrid" in opts.techniques:
        return f"Hybrid {name} · {search}"
    return f"Dense-only {name} · dense vector search"


def catalog(available_rerankers: dict[str, bool]) -> dict:
    return {
        "modes": [{"id": m, "label": MODE_INFO[m][0], "description": MODE_INFO[m][1]} for m in MODES],
        "lightrag_modes": [{"id": m, "label": m.capitalize(), "description": LIGHTRAG_INFO[m]} for m in LIGHTRAG_MODES],
        "techniques": [{"id": t, "label": v[0], "description": v[1]} for t, v in TECHNIQUE_INFO.items()],
        "rerankers": [{"id": r, "label": v[0], "description": v[1], "available": available_rerankers.get(r, False)}
                      for r, v in RERANKER_INFO.items()],
        "hybrid_labels": {m: hybrid_label(Options(m, "hybrid", frozenset({"hybrid"}), ("cross_encoder",)))
                          for m in MODES},
        "dense_labels": {m: hybrid_label(Options(m, "hybrid", frozenset(), ("cross_encoder",)))
                         for m in MODES},
    }
