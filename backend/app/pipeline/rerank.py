"""Rerankers. The cross-encoder is the default. Other options can be combined; when several are chosen,
their orderings are fused by reciprocal rank fusion. The refusal threshold uses cross-encoder scores only,
since other rerankers use different scales."""

import importlib.util
import logging

import numpy as np

from ..config import get_settings
from ..parsing.types import ParseError

RRF_K = 60
log = logging.getLogger("rerank")


class RerankerHub:
    def __init__(self, embedder, cross_encoder):
        self.embedder = embedder
        self.cross = cross_encoder
        self._flashrank = None

    def available(self) -> dict[str, bool]:
        s = get_settings()
        return {
            "cross_encoder": self.cross is not None,
            "bi_encoder": True,
            "flashrank": importlib.util.find_spec("flashrank") is not None,
            "cohere": bool(s.cohere_api_key) and importlib.util.find_spec("cohere") is not None,
        }

    def score(self, name: str, query: str, chunks: list[dict]) -> list[float]:
        texts = [c["text"] for c in chunks]
        if name == "cross_encoder":
            if self.cross is None:
                # Only when the index was built without a reranker (tests, some evaluation runs).
                log.warning("cross-encoder not loaded; scoring with bi-encoder instead")
                return self.score("bi_encoder", query, chunks)
            return [float(x) for x in self.cross.predict([(query, t) for t in texts])]
        if name == "bi_encoder":
            q = np.asarray(self.embedder.encode(query, normalize_embeddings=True), dtype=np.float32)
            P = np.asarray(self.embedder.encode(texts, normalize_embeddings=True), dtype=np.float32)
            return [float(x) for x in P @ q]
        if name == "flashrank":
            return self._flashrank_scores(query, texts)
        if name == "cohere":
            return self._cohere_scores(query, texts)
        raise ValueError(f"unknown reranker: {name}")

    def _flashrank_scores(self, query: str, texts: list[str]) -> list[float]:
        try:
            from flashrank import Ranker, RerankRequest
        except ImportError:
            raise ParseError("FlashRank is not installed. Install the flashrank package, or choose another reranker.")
        if self._flashrank is None:
            self._flashrank = Ranker()
        request = RerankRequest(query=query, passages=[{"id": i, "text": t} for i, t in enumerate(texts)])
        scores = [0.0] * len(texts)
        for result in self._flashrank.rerank(request):
            scores[int(result["id"])] = float(result["score"])
        return scores

    def _cohere_scores(self, query: str, texts: list[str]) -> list[float]:
        s = get_settings()
        if not s.cohere_api_key:
            raise ParseError("Cohere Rerank needs COHERE_API_KEY. Set it, or choose another reranker.")
        try:
            import cohere
        except ImportError:
            raise ParseError("The cohere package is not installed.")
        client = cohere.ClientV2(api_key=s.cohere_api_key)
        response = client.rerank(model=s.cohere_rerank_model, query=query, documents=texts, top_n=len(texts))
        scores = [0.0] * len(texts)
        for result in response.results:
            scores[result.index] = float(result.relevance_score)
        return scores

    def rank(self, names: tuple, query: str, chunks: list[dict]) -> list[tuple[dict, float | None, float]]:
        """Returns (chunk, gate_score, rank_score), best first. gate_score is the cross-encoder score,
        or None when the cross-encoder isn't part of the ranking."""
        if not chunks:
            return []
        per_name = {n: self.score(n, query, chunks) for n in names}
        cross = per_name.get("cross_encoder")
        if len(names) == 1:
            scores = per_name[names[0]]
            order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
            return [(chunks[i], scores[i] if names[0] == "cross_encoder" else None, scores[i]) for i in order]

        fused = [0.0] * len(chunks)
        for n in names:
            order = sorted(range(len(chunks)), key=lambda i: per_name[n][i], reverse=True)
            for rank, i in enumerate(order):
                fused[i] += 1.0 / (RRF_K + rank + 1)
        order = sorted(range(len(chunks)), key=lambda i: fused[i], reverse=True)
        return [(chunks[i], cross[i] if cross is not None else None, fused[i]) for i in order]


def hub_for(index) -> RerankerHub:
    """One hub per index, so the embedder and cross-encoder are loaded once."""
    hub = getattr(index, "_hub", None)
    if hub is None:
        hub = RerankerHub(index.embedder, index.reranker)
        index._hub = hub
    return hub
