"""The shared retrieval path used by every mode: build a candidate pool (optionally rewritten queries, dense
or hybrid search, graph chunks), rank it with the chosen rerankers, apply the expansion techniques, and gate
on the refusal threshold."""

from ..config import get_settings
from ..retrieval import RRF_K
from .options import Options
from .rerank import RerankerHub
from .techniques import auto_merge, compress, parent_expand, sentence_window


def _fuse_chunk_lists(lists: list[list[dict]]) -> list[dict]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranked in lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk["id"]] = scores.get(chunk["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
            by_id.setdefault(chunk["id"], chunk)
    return [by_id[cid] for cid in sorted(scores, key=scores.get, reverse=True)]


def build_pool(index, queries: list[str], opts: Options, extra_chunk_ids: list[str] = ()) -> list[dict]:
    s = get_settings()
    search_mode = "hybrid" if "hybrid" in opts.techniques else "dense"
    lists = [index.search(q, mode=search_mode, rerank=False, k=s.dense_k, candidates=s.dense_k) for q in queries]
    pool = _fuse_chunk_lists(lists)[: s.dense_k * 2]
    seen = {c["id"] for c in pool}
    for chunk in index.get_chunks(list(extra_chunk_ids)):
        if chunk["id"] not in seen:
            pool.append(chunk)
            seen.add(chunk["id"])
    return pool


def finish(hub: RerankerHub, question: str, pool: list[dict], opts: Options) -> list[dict]:
    """Rerank, expand with the selected techniques, and drop passages below the refusal threshold."""
    s = get_settings()
    ranked = hub.rank(opts.rerankers, question, pool)[: s.rerank_top_n * 2]
    passages = [{**c, "score": rank_score, "gate": gate} for c, gate, rank_score in ranked]

    if "auto_merging" in opts.techniques:
        passages = auto_merge(passages)
    elif "parent_document" in opts.techniques:
        passages = parent_expand(passages)

    if "contextual_compression" in opts.techniques:
        passages = compress(passages, question, hub, opts.rerankers)
    elif "sentence_window" in opts.techniques:
        passages = sentence_window(passages, question, hub, opts.rerankers)

    passages = passages[: s.rerank_top_n]
    return [p for p in passages if p["gate"] is None or p["gate"] >= s.rerank_min_score]
