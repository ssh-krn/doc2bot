"""Chunk index: dense vectors (Qdrant) + BM25, fused with reciprocal rank fusion, then reranked.
The knowledge graph (app/kg) uses this index for chunk lookup and reranking."""

import re
import uuid
from threading import Lock

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+")
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _rrf(rankings: list[list[str]]) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Index:
    def __init__(self, embedder, client, collection: str, dim: int, reranker=None):
        self.embedder = embedder
        self.client = client
        self.collection = collection
        self.reranker = reranker
        self.lock = Lock()  # local Qdrant and BM25 are not thread-safe
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None
        if not client.collection_exists(collection):
            client.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        self._load_chunks()

    def _load_chunks(self) -> None:
        offset, chunks = None, []
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            chunks.extend({"id": str(p.id), **p.payload} for p in points)
            if offset is None:
                break
        self.chunks = chunks
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks]) if self.chunks else None

    def add_document(self, doc_id: str, filename: str, chunks: list[dict]) -> list[str]:
        """Indexes chunks and returns their ids, in the same order as `chunks`."""
        if not chunks:
            return []
        vectors = self.embedder.encode(
            [c["text"] for c in chunks], normalize_embeddings=True, batch_size=32
        )
        points, new, ids = [], [], []
        for c, v in zip(chunks, vectors):
            pid = str(uuid.uuid4())
            payload = {
                "text": c["text"],
                "doc_id": doc_id,
                "filename": filename,
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "modality": c.get("modality", "text"),
                "image_key": c.get("image_key") or "",
                "parent_id": c.get("parent_id") or f"self-{pid}",
                "parent_text": c.get("parent_text") or c["text"],
            }
            points.append(PointStruct(id=pid, vector=[float(x) for x in v], payload=payload))
            new.append({"id": pid, **payload})
            ids.append(pid)
        with self.lock:
            self.client.upsert(self.collection, points=points)
            self.chunks.extend(new)
            self._rebuild_bm25()
        return ids

    def delete_document(self, doc_id: str) -> None:
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        with self.lock:
            self.client.delete(self.collection, points_selector=FilterSelector(filter=flt))
            self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
            self._rebuild_bm25()

    def chunk_ids_for_doc(self, doc_id: str) -> list[str]:
        with self.lock:
            return [c["id"] for c in self.chunks if c["doc_id"] == doc_id]

    def image_keys_for_doc(self, doc_id: str) -> list[str]:
        with self.lock:
            return [c["image_key"] for c in self.chunks if c["doc_id"] == doc_id and c.get("image_key")]

    def get_chunks(self, ids: list[str]) -> list[dict]:
        with self.lock:
            by_id = {c["id"]: c for c in self.chunks}
        return [by_id[i] for i in ids if i in by_id]

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(
                c["doc_id"],
                {"doc_id": c["doc_id"], "filename": c["filename"], "chunks": 0, "pages": set()},
            )
            d["chunks"] += 1
            d["pages"].add(c["page"])
        return [
            {
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "kind": d["filename"].rsplit(".", 1)[-1].lower() if "." in d["filename"] else "",
                "chunks": d["chunks"],
                "pages": len(d["pages"]),
            }
            for d in docs.values()
        ]

    def rerank(self, query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
        """Cross-encoder scores for chunks, best first. Without a reranker, keeps the input order."""
        if not chunks:
            return []
        if self.reranker is None:
            return [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
        raw = self.reranker.predict([(query, c["text"]) for c in chunks])
        return sorted(zip(chunks, (float(x) for x in raw)), key=lambda x: x[1], reverse=True)

    def search(
        self,
        query: str,
        mode: str = "hybrid",       # "dense", "bm25", or "hybrid"
        rerank: bool = True,
        k: int = 5,
        candidates: int = 20,
    ) -> list[dict]:
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"unknown mode: {mode}")
        with self.lock:
            if not self.chunks:
                return []
            dense_ids: list[str] = []
            bm25_ids: list[str] = []
            if mode in ("dense", "hybrid"):
                qv = [float(x) for x in self.embedder.encode(query, normalize_embeddings=True)]
                hits = self.client.query_points(
                    self.collection, query=qv, limit=candidates, with_payload=False
                ).points
                dense_ids = [str(h.id) for h in hits]
            if mode in ("bm25", "hybrid") and self.bm25 is not None:
                scores = self.bm25.get_scores(tokenize(query))
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                bm25_ids = [self.chunks[i]["id"] for i in order[:candidates] if scores[i] > 0]
            by_id = {c["id"]: c for c in self.chunks}

        if mode == "hybrid":
            ranked_ids = _rrf([dense_ids, bm25_ids])
        elif mode == "dense":
            ranked_ids = dense_ids
        else:
            ranked_ids = bm25_ids

        pool = [by_id[i] for i in ranked_ids[:candidates] if i in by_id]
        if not pool:
            return []
        if rerank and self.reranker is not None:
            scored = self.rerank(query, pool)
            return [{**c, "score": s, "reranked": True} for c, s in scored[:k]]
        return [{**c, "score": 1.0 / (rank + 1), "reranked": False} for rank, c in enumerate(pool[:k])]
