"""Evaluate retrieval ablations and (optionally) end-to-end answers.

Run from backend/:
  python -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --chunk-sizes 400 800 1200
  python -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --answer --answer-chunk-size 800

Retrieval ablations need no LLM calls. --answer calls the LLM once per question.
Cost is computed from PRICE_IN_PER_M / PRICE_OUT_PER_M (USD per million tokens) env vars,
which you should set from your provider's current pricing page.
"""

import argparse
import json
import os
import re
import statistics
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingest import extract_chunks
from app.retrieval import Index

from .common import is_gold_chunk, load_qa, norm, percentile

RESULTS = Path("eval/results")
CONFIGS = [("dense", False), ("bm25", False), ("hybrid", False), ("dense", True), ("hybrid", True)]
REFUSAL_MARKERS = ["couldn't find", "could not find", "don't have enough information", "do not have enough information"]


def build_index(files, chunk_size, embedder, reranker, dim) -> Index:
    s = get_settings()
    overlap = int(chunk_size * s.chunk_overlap / s.chunk_size)
    idx = Index(embedder, QdrantClient(":memory:"), f"chunks_{chunk_size}", dim, reranker)
    for f in files:
        chunks, _ = extract_chunks(f.read_bytes(), chunk_size, overlap)
        idx.add_document(str(uuid.uuid4()), f.name, chunks)
    return idx


def label_coverage(idx: Index, qa_items: list[dict]) -> float:
    """Share of questions whose gold snippet exists in at least one chunk.
    Below 100% means the chunker split the answer, so no retriever can find it."""
    hits = sum(any(norm(q["gold_snippet"]) in norm(c["text"]) for c in idx.chunks) for q in qa_items)
    return hits / len(qa_items) if qa_items else 0.0


def run_retrieval(idx: Index, qa_items: list[dict], mode: str, rerank: bool, ks=(1, 3, 5)) -> dict:
    ranks, latencies = [], []
    import time
    for qa in qa_items:
        t0 = time.perf_counter()
        hits = idx.search(qa["question"], mode=mode, rerank=rerank, k=max(ks), candidates=20)
        latencies.append((time.perf_counter() - t0) * 1000)
        found = [i + 1 for i, h in enumerate(hits) if is_gold_chunk(h, qa)]
        ranks.append(found[0] if found else None)
    n = len(qa_items)
    out = {f"recall@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks}
    out["mrr@5"] = sum(1 / r for r in ranks if r and r <= 5) / n
    out["p50_ms"] = percentile(latencies, 50)
    out["p95_ms"] = percentile(latencies, 95)
    return out


def to_markdown(rows: list[dict]) -> str:
    head = (
        "| Chunk size | Mode | Rerank | Recall@1 | Recall@3 | Recall@5 | MRR@5 | p50 ms | p95 ms | Label coverage |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = [
        f"| {r['chunk_size']} | {r['mode']} | {'yes' if r['rerank'] else 'no'} | "
        f"{r['recall@1']:.0%} | {r['recall@3']:.0%} | {r['recall@5']:.0%} | {r['mrr@5']:.2f} | "
        f"{r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['label_coverage']:.0%} |"
        for r in rows
    ]
    return head + "\n".join(body) + "\n"


def run_answers(idx: Index, qa_items: list[dict]) -> list[dict]:
    from app.rag import answer_once  # lazy import: needs LLM dependencies and an API key

    results = []
    for qa in qa_items:
        answerable = qa.get("answerable", True)
        out = answer_once(idx, qa["question"])
        ans, srcs = out["answer"], out["sources"]
        low = ans.lower()
        refused = any(m in low for m in REFUSAL_MARKERS)

        if answerable:
            correct = all(norm(k) in norm(ans) for k in qa.get("gold_answer_keywords", [])) and not refused
        else:
            correct = refused

        def src_is_gold(s):
            return answerable and s["filename"] == qa["gold_doc"] and s["page"] == qa["gold_page"] \
                and norm(qa["gold_snippet"]) in norm(s["text"])

        cited = [int(n) for n in re.findall(r"\[(\d+)\]", ans) if 1 <= int(n) <= len(srcs)]
        bad_cites = sum(1 for n in cited if not src_is_gold(srcs[n - 1]))
        results.append({
            "id": qa["id"],
            "answerable": answerable,
            "correct": correct,
            "refused": refused,
            "retrieved_gold": any(src_is_gold(s) for s in srcs),
            "n_cites": len(cited),
            "bad_cites": bad_cites,
            "retrieval_ms": out["retrieval_ms"],
            "llm_ms": out["llm_ms"],
            "input_tokens": out["input_tokens"],
            "output_tokens": out["output_tokens"],
            "answer": ans,
        })
    return results


def classify_failure(r: dict, chunk_exists: bool) -> str | None:
    if not r["answerable"]:
        return None if r["refused"] else "hallucinated_answer"
    if not r["retrieved_gold"]:
        return "retrieval_miss" if chunk_exists else "snippet_split_by_chunking"
    if r["refused"]:
        return "false_refusal"
    if not r["correct"]:
        return "wrong_answer"
    if r["bad_cites"]:
        return "bad_citation"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[400, 800, 1200])
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--answer-chunk-size", type=int, default=800)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    files = sorted(Path(args.corpus).glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs found in {args.corpus}")
    qa_all = load_qa(args.qa)
    answerable = [q for q in qa_all if q.get("answerable", True)]
    if not answerable:
        raise SystemExit("No answerable questions in the QA file")

    from sentence_transformers import CrossEncoder, SentenceTransformer
    embedder = SentenceTransformer(s.embed_model)
    reranker = CrossEncoder(s.rerank_model)
    dim = embedder.get_sentence_embedding_dimension()

    rows, indexes = [], {}
    for cs in args.chunk_sizes:
        idx = build_index(files, cs, embedder, reranker, dim)
        indexes[cs] = idx
        cov = label_coverage(idx, answerable)
        for mode, rerank in CONFIGS:
            m = run_retrieval(idx, answerable, mode, rerank)
            rows.append({"chunk_size": cs, "mode": mode, "rerank": rerank, "label_coverage": cov, **m})
            print(f"chunk={cs} mode={mode} rerank={rerank} recall@5={m['recall@5']:.0%} p50={m['p50_ms']:.0f}ms")

    (RESULTS / "ablation.json").write_text(json.dumps(rows, indent=2))
    (RESULTS / "ablation.md").write_text(to_markdown(rows))
    print(f"\nWrote {RESULTS/'ablation.md'}")

    if args.answer:
        cs = args.answer_chunk_size
        idx = indexes.get(cs) or build_index(files, cs, embedder, reranker, dim)
        chunk_exists = {q["id"]: any(norm(q["gold_snippet"]) in norm(c["text"]) for c in idx.chunks) for q in qa_all}
        results = run_answers(idx, qa_all)
        qa_by_id = {q["id"]: q for q in qa_all}

        n = len(results)
        ans_items = [r for r in results if r["answerable"]]
        unans = [r for r in results if not r["answerable"]]
        in_tok = [r["input_tokens"] for r in results]
        out_tok = [r["output_tokens"] for r in results]
        price_in = float(os.getenv("PRICE_IN_PER_M", "0"))
        price_out = float(os.getenv("PRICE_OUT_PER_M", "0"))
        cost_per_q = (statistics.mean(in_tok) * price_in + statistics.mean(out_tok) * price_out) / 1e6

        all_cites = sum(r["n_cites"] for r in results)
        summary = {
            "chunk_size": cs,
            "questions": n,
            "answer_accuracy_answerable": sum(r["correct"] for r in ans_items) / max(1, len(ans_items)),
            "refusal_accuracy_unanswerable": sum(r["correct"] for r in unans) / max(1, len(unans)),
            "citation_error_rate": sum(r["bad_cites"] for r in results) / max(1, all_cites),
            "retrieval_p50_ms": percentile([r["retrieval_ms"] for r in results], 50),
            "llm_p50_ms": percentile([r["llm_ms"] for r in results], 50),
            "llm_p95_ms": percentile([r["llm_ms"] for r in results], 95),
            "mean_input_tokens": statistics.mean(in_tok),
            "mean_output_tokens": statistics.mean(out_tok),
            "cost_per_query_usd": cost_per_q,
            "cost_per_1000_queries_usd": cost_per_q * 1000,
        }
        (RESULTS / "answer_eval.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))

        with open(RESULTS / "failures.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                cat = classify_failure(r, chunk_exists[r["id"]])
                if cat:
                    q = qa_by_id[r["id"]]
                    f.write(json.dumps({"id": r["id"], "category": cat, "question": q["question"],
                                        "gold": q.get("gold_doc"), "page": q.get("gold_page"),
                                        "answer": r["answer"]}, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
