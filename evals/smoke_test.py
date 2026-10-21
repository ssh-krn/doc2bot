"""Offline smoke test: synthetic PDF, stub embedder, no reranker. Checks ingestion,
hybrid/BM25/dense retrieval, labeling, and metrics without downloading models.
Run from backend/: python -m eval.smoke_test"""

import json
import tempfile
import uuid
import zlib
from pathlib import Path

import fitz
import numpy as np
from qdrant_client import QdrantClient

from app.retrieval import Index, tokenize
from app.ingest import extract_chunks
from eval.common import is_gold_chunk
from eval.run_eval import run_retrieval, label_coverage

DIM = 64


class StubEmbedder:
    """Bag-of-words hashing embedder. Stands in for sentence-transformers in tests only."""

    def get_sentence_embedding_dimension(self):
        return DIM

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        out = np.zeros((len(texts), DIM))
        for i, t in enumerate(texts):
            for w in tokenize(t):
                out[i, zlib.crc32(w.encode()) % DIM] += 1
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out[0] if single else out


def main():
    tmp = Path(tempfile.mkdtemp())
    pdf = fitz.open()
    pages = [
        "The refund policy allows returns within 30 days of purchase with a receipt.",
        "Shipping to remote regions takes 10 to 14 business days and costs extra.",
        "Warranty claims require the original invoice and the device serial number.",
    ]
    for text in pages:
        pdf.new_page().insert_text((72, 72), text)
    (tmp / "policy.pdf").write_bytes(pdf.tobytes())
    pdf.close()

    qa = [
        {"id": "q1", "question": "How long do I have to return an item?", "gold_doc": "policy.pdf",
         "gold_page": 1, "gold_snippet": "returns within 30 days of purchase", "gold_answer_keywords": ["30 days"]},
        {"id": "q2", "question": "How long does shipping to remote regions take?", "gold_doc": "policy.pdf",
         "gold_page": 2, "gold_snippet": "takes 10 to 14 business days", "gold_answer_keywords": ["10 to 14"]},
        {"id": "q3", "question": "What do warranty claims require?", "gold_doc": "policy.pdf",
         "gold_page": 3, "gold_snippet": "original invoice and the device serial number", "gold_answer_keywords": ["invoice"]},
    ]
    qa_file = tmp / "qa.jsonl"
    qa_file.write_text("\n".join(json.dumps(q) for q in qa), encoding="utf-8")

    emb = StubEmbedder()
    chunks, page_count = extract_chunks((tmp / "policy.pdf").read_bytes(), 200, 30)
    assert page_count == 3, page_count
    assert len(chunks) >= 3, chunks

    idx = Index(emb, QdrantClient(":memory:"), "test", DIM, reranker=None)
    idx.add_document(str(uuid.uuid4()), "policy.pdf", chunks)
    assert idx.list_documents()[0]["pages"] == 3

    assert label_coverage(idx, qa) == 1.0
    for mode in ["dense", "bm25", "hybrid"]:
        m = run_retrieval(idx, qa, mode, rerank=False)
        print(f"{mode:7s} recall@1={m['recall@1']:.2f} recall@3={m['recall@3']:.2f} mrr@5={m['mrr@5']:.2f}")
        if mode in ("dense", "hybrid"):
            assert m["recall@3"] == 1.0, f"{mode} missed a gold passage in top 3"
        # BM25 alone is expected to miss "require" vs "requires" (no stemming): a real ablation signal.

    hits = idx.search("How long do I have to return an item?", mode="hybrid", rerank=True, k=3)
    assert hits and hits[0]["reranked"] is False  # no reranker injected, so it must report that honestly
    assert any(is_gold_chunk(h, qa[0]) for h in hits), [h["text"] for h in hits]  # gold within top 3

    idx.delete_document(idx.list_documents()[0]["doc_id"])
    assert idx.list_documents() == []
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
