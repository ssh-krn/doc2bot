"""Evaluation harness: retrieval ablations, end-to-end answers, mode comparison and failure classification.

Run from backend/. The corpus and questions come from eval/build_corpus.py and eval/build_qa.py.

  # retrieval only, no models needed (BM25 keyword search):
  python -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --lexical-only

  # retrieval with the embedding and reranking models (downloads them once):
  python -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --chunk-sizes 400 800 1200

  # answers: one mode, or several to compare (calls the chat model once per question per mode):
  python -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --answer --answer-mode crag
  python -m eval.run_eval --corpus eval/corpus --qa eval/qa.jsonl --compare-modes traditional crag lightrag

Metrics:
  recall@k      share of answerable questions whose gold passages are ALL in the top k (multi-hop needs every hop)
  MRR@5         mean reciprocal rank of the first gold passage, within the top 5
  label coverage share of questions whose gold snippets exist in some chunk. Below 100% means chunking split an answer.
  accuracy      answerable: every answer keyword present and no refusal. unanswerable: the system refused.
  citation err  citations whose source does not contain the gold passage, out of all citations
Cost comes from PRICE_IN_PER_M and PRICE_OUT_PER_M (USD per million tokens) in the environment.
"""

import argparse
import json
import os
import re
import statistics
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingest import extract_chunks
from app.retrieval import Index

from .common import all_golds_found, load_qa, matches, norm, percentile

RESULTS = Path("eval/results")
CONFIGS = [("dense", False), ("bm25", False), ("hybrid", False), ("dense", True), ("hybrid", True)]
LEXICAL_CONFIGS = [("bm25", False)]
REFUSAL_MARKERS = ["couldn't find", "could not find", "don't have enough information", "do not have enough information"]
QTYPES = ["fact", "definition", "comparison", "numeric", "multi_hop"]


# ---------- indexing ----------

def build_index(files, chunk_size, embedder, reranker, dim) -> Index:
    """embedder=None builds a lexical-only index (BM25)."""
    s = get_settings()
    overlap = int(chunk_size * s.chunk_overlap / s.chunk_size)
    idx = Index(embedder, QdrantClient(":memory:"), f"chunks_{chunk_size}", dim, reranker)
    for f in files:
        chunks, _ = extract_chunks(f.read_bytes(), chunk_size, overlap)
        idx.add_document(str(uuid.uuid4()), f.name, chunks)
    return idx


def label_coverage(idx: Index, items: list[dict]) -> float:
    if not items:
        return 0.0
    ok = sum(all(any(matches(c, g) for c in idx.chunks) for g in qa["golds"]) for qa in items)
    return ok / len(items)


# ---------- retrieval ----------

def run_retrieval_fn(search, items: list[dict], ks=(1, 3, 5)) -> dict:
    """search(question) -> ranked chunk dicts. Returns recall@k, MRR@5, latency, and recall@5 per question type."""
    found: dict[int, list[bool]] = {k: [] for k in ks}
    rr: list[float] = []
    latencies: list[float] = []
    by_type: dict[str, list[bool]] = {}
    for qa in items:
        t0 = time.perf_counter()
        hits = search(qa["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        for k in ks:
            found[k].append(all_golds_found(hits[:k], qa))
        first = next((i + 1 for i, h in enumerate(hits[:5]) if any(matches(h, g) for g in qa["golds"])), None)
        rr.append(1 / first if first else 0.0)
        by_type.setdefault(qa["type"], []).append(all_golds_found(hits[:5], qa))
    n = max(1, len(items))
    out = {f"recall@{k}": sum(found[k]) / n for k in ks}
    out["mrr@5"] = sum(rr) / n
    out["p50_ms"] = percentile(latencies, 50)
    out["p95_ms"] = percentile(latencies, 95)
    out["by_type"] = {t: sum(v) / len(v) for t, v in by_type.items()}
    out["questions"] = len(items)
    return out


def run_retrieval(idx: Index, items: list[dict], mode: str, rerank: bool, ks=(1, 3, 5)) -> dict:
    return run_retrieval_fn(lambda q: idx.search(q, mode=mode, rerank=rerank, k=max(ks), candidates=20), items, ks)


# ---------- answers ----------

def run_answers(idx: Index, items: list[dict], graph=None, mode: str = "traditional", techniques=None,
                rerankers=None, lightrag_mode: str | None = None) -> list[dict]:
    from app.rag import answer_once  # imported here: needs LLM dependencies and an API key

    results = []
    for qa in items:
        answerable = qa["answerable"]
        out = answer_once(idx, graph, qa["question"], mode, techniques=techniques, rerankers=rerankers,
                          lightrag_mode=lightrag_mode)
        ans, srcs = out["answer"], out["sources"]
        low = ans.lower()
        refused = any(m in low for m in REFUSAL_MARKERS)
        if answerable:
            correct = all(norm(k) in norm(ans) for k in qa["gold_answer_keywords"]) and not refused
        else:
            correct = refused

        def supports(s: dict) -> bool:
            return answerable and any(matches(s, g) for g in qa["golds"])

        cited = [int(n) for n in re.findall(r"\[(\d+)\]", ans) if 1 <= int(n) <= len(srcs)]
        results.append({
            "id": qa["id"],
            "type": qa["type"],
            "answerable": answerable,
            "correct": correct,
            "refused": refused,
            "retrieved_gold": answerable and all_golds_found(srcs, qa),
            "n_cites": len(cited),
            "bad_cites": sum(1 for n in cited if not supports(srcs[n - 1])),
            "retrieval_ms": out["retrieval_ms"],
            "llm_ms": out["llm_ms"],
            "input_tokens": out["input_tokens"],
            "output_tokens": out["output_tokens"],
            "answer": ans,
            "notes": out.get("notes", []),
        })
    return results


def summarize(results: list[dict], chunk_size: int, mode: str) -> dict:
    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]
    in_tok = [r["input_tokens"] for r in results] or [0]
    out_tok = [r["output_tokens"] for r in results] or [0]
    price_in = float(os.getenv("PRICE_IN_PER_M", "0"))
    price_out = float(os.getenv("PRICE_OUT_PER_M", "0"))
    cost = (statistics.mean(in_tok) * price_in + statistics.mean(out_tok) * price_out) / 1e6
    all_cites = sum(r["n_cites"] for r in results)
    by_type = {}
    for t in sorted({r["type"] for r in ans}):
        rows = [r for r in ans if r["type"] == t]
        by_type[t] = sum(r["correct"] for r in rows) / len(rows)
    return {
        "mode": mode,
        "chunk_size": chunk_size,
        "questions": len(results),
        "answer_accuracy_answerable": sum(r["correct"] for r in ans) / max(1, len(ans)),
        "refusal_accuracy_unanswerable": sum(r["correct"] for r in unans) / max(1, len(unans)),
        "answer_accuracy_by_type": by_type,
        "citation_error_rate": sum(r["bad_cites"] for r in results) / max(1, all_cites),
        "retrieval_p50_ms": percentile([r["retrieval_ms"] for r in results], 50),
        "llm_p50_ms": percentile([r["llm_ms"] for r in results], 50),
        "llm_p95_ms": percentile([r["llm_ms"] for r in results], 95),
        "mean_input_tokens": statistics.mean(in_tok),
        "mean_output_tokens": statistics.mean(out_tok),
        "cost_per_query_usd": cost,
        "cost_per_1000_queries_usd": cost * 1000,
    }


def classify_failure(r: dict, labels_exist: bool) -> str | None:
    if not r["answerable"]:
        return None if r["refused"] else "hallucinated_answer"
    if not r["retrieved_gold"]:
        return "retrieval_miss" if labels_exist else "snippet_split_by_chunking"
    if r["refused"]:
        return "false_refusal"
    if not r["correct"]:
        return "wrong_answer"
    if r["bad_cites"]:
        return "bad_citation"
    return None


# ---------- reporting ----------

def ablation_markdown(rows: list[dict]) -> str:
    head = ("| Chunk size | Method | Rerank | Recall@1 | Recall@3 | Recall@5 | MRR@5 | p50 ms | p95 ms | Label coverage |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    body = [
        f"| {r['chunk_size']} | {r['mode']} | {'yes' if r['rerank'] else 'no'} | {r['recall@1']:.0%} | "
        f"{r['recall@3']:.0%} | {r['recall@5']:.0%} | {r['mrr@5']:.2f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | "
        f"{r['label_coverage']:.0%} |"
        for r in rows
    ]
    types_head = ("\n\nRecall@5 by question type:\n\n| Chunk size | Method | " + " | ".join(QTYPES) + " |\n"
                  "|---|---|" + "---|" * len(QTYPES) + "\n")
    types_body = [
        f"| {r['chunk_size']} | {r['mode']} | " + " | ".join(
            f"{r['by_type'][t]:.0%}" if t in r["by_type"] else "–" for t in QTYPES) + " |"
        for r in rows
    ]
    return head + "\n".join(body) + "\n" + types_head + "\n".join(types_body) + "\n"


def compare_markdown(summaries: list[dict]) -> str:
    head = ("| Mode | Answer accuracy | Refusal accuracy | Citation error | Retrieval p50 ms | Model p50 ms | "
            "Tokens in / out (mean) |\n|---|---|---|---|---|---|---|\n")
    body = [
        f"| {s['mode']} | {s['answer_accuracy_answerable']:.0%} | {s['refusal_accuracy_unanswerable']:.0%} | "
        f"{s['citation_error_rate']:.0%} | {s['retrieval_p50_ms']:.0f} | {s['llm_p50_ms']:.0f} | "
        f"{s['mean_input_tokens']:.0f} / {s['mean_output_tokens']:.0f} |"
        for s in summaries
    ]
    return head + "\n".join(body) + "\n"


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="eval/corpus")
    ap.add_argument("--qa", default="eval/qa.jsonl")
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[400, 800, 1200])
    ap.add_argument("--lexical-only", action="store_true", help="BM25 retrieval only; no models, no answers")
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--answer-mode", default="traditional")
    ap.add_argument("--compare-modes", nargs="+", default=None, help="answer with each mode and compare")
    ap.add_argument("--answer-chunk-size", type=int, default=800)
    ap.add_argument("--lightrag-mode", default="hybrid", choices=["local", "global", "hybrid"])
    ap.add_argument("--techniques", nargs="*", default=["hybrid"])
    ap.add_argument("--rerankers", nargs="*", default=["cross_encoder"])
    ap.add_argument("--kg", action="store_true", help="build a graph (one LLM call per chunk) and measure graph modes")
    ap.add_argument("--kg-chunk-size", type=int, default=800)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    files = sorted(Path(args.corpus).glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs in {args.corpus}. Run: python -m eval.build_corpus")
    items = load_qa(args.qa)
    answerable = [q for q in items if q["answerable"]]
    if not answerable:
        raise SystemExit("No answerable questions in the QA file")
    print(f"{len(files)} documents, {len(items)} questions ({len(answerable)} answerable)")

    embedder = reranker = None
    dim = 1
    if not args.lexical_only:
        from sentence_transformers import CrossEncoder, SentenceTransformer
        embedder = SentenceTransformer(s.embed_model)
        reranker = CrossEncoder(s.rerank_model)
        dim = embedder.get_sentence_embedding_dimension()
    configs = LEXICAL_CONFIGS if args.lexical_only else CONFIGS

    rows, indexes, graph = [], {}, None
    for cs in args.chunk_sizes:
        idx = build_index(files, cs, embedder, reranker, dim)
        indexes[cs] = idx
        cov = label_coverage(idx, answerable)
        for mode, rerank in configs:
            m = run_retrieval(idx, answerable, mode, rerank)
            rows.append({"chunk_size": cs, "mode": mode, "rerank": rerank, "label_coverage": cov, **m})
            print(f"chunk={cs} {mode:6s} rerank={str(rerank):5s} recall@5={m['recall@5']:.0%} "
                  f"MRR@5={m['mrr@5']:.2f} p50={m['p50_ms']:.1f}ms")

    if args.kg:
        from app.kg.build import index_document, llm_invoke_factory
        from app.kg.graph import KnowledgeGraph
        kidx = indexes.get(args.kg_chunk_size) or build_index(files, args.kg_chunk_size, embedder, reranker, dim)
        graph = KnowledgeGraph(embedder, path=None)  # in-memory; the app's saved graph is never touched
        invoke = llm_invoke_factory()
        for doc_id in sorted({c["doc_id"] for c in kidx.chunks}):
            doc_chunks = [{"id": c["id"], "text": c["text"]} for c in kidx.chunks if c["doc_id"] == doc_id]
            index_document(graph, invoke, doc_id, doc_chunks, workers=s.kg_workers)
        print("graph stats:", graph.stats())
        cov = label_coverage(kidx, answerable)
        for kg_mode in ["local", "global", "hybrid"]:
            m = run_retrieval_fn(lambda q, mode=kg_mode: _lightrag_search(kidx, graph, q, mode), answerable)
            rows.append({"chunk_size": args.kg_chunk_size, "mode": f"kg-{kg_mode}", "rerank": True,
                         "label_coverage": cov, **m})
            print(f"kg-{kg_mode}: recall@5={m['recall@5']:.0%}")

    (RESULTS / "ablation.json").write_text(json.dumps(rows, indent=2))
    (RESULTS / "ablation.md").write_text(ablation_markdown(rows))
    print(f"wrote {RESULTS / 'ablation.md'}")

    if args.lexical_only:
        print("lexical-only run: answers and dense/graph retrieval were not measured.")
        return
    if not (args.answer or args.compare_modes):
        return

    cs = args.answer_chunk_size
    idx = indexes.get(cs) or build_index(files, cs, embedder, reranker, dim)
    modes = args.compare_modes or [args.answer_mode]
    summaries, primary = [], None
    for mode in modes:
        print(f"answering {len(items)} questions with mode={mode} ...")
        if mode in ("lightrag", "graphrag") and graph is None:
            print(f"note: {mode} needs --kg (a built graph); without it the mode falls back to chunk search")
        results = run_answers(idx, items, graph, mode, techniques=args.techniques, rerankers=args.rerankers,
                              lightrag_mode=args.lightrag_mode)
        summary = summarize(results, cs, mode)
        summaries.append(summary)
        if primary is None:
            primary = (summary, results)
        labels_exist = {q["id"]: all(any(matches(c, g) for c in idx.chunks) for g in q["golds"]) for q in items}
        with open(RESULTS / f"failures_{mode}.jsonl", "w", encoding="utf-8") as f:
            qa_by_id = {q["id"]: q for q in items}
            for r in results:
                cat = classify_failure(r, labels_exist[r["id"]])
                if cat:
                    q = qa_by_id[r["id"]]
                    f.write(json.dumps({"id": r["id"], "type": q["type"], "category": cat, "question": q["question"],
                                        "answer": r["answer"]}, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2))

    summary, results = primary
    (RESULTS / "answer_eval.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    if len(summaries) > 1:
        (RESULTS / "compare.json").write_text(json.dumps(summaries, indent=2))
        (RESULTS / "compare.md").write_text(compare_markdown(summaries))
        print(compare_markdown(summaries))


def _lightrag_search(idx, graph, question, kg_mode):
    from app.rag import retrieve
    from app.pipeline.options import make_options

    return retrieve(idx, graph, question, "lightrag", make_options("lightrag", lightrag_mode=kg_mode))[0]


if __name__ == "__main__":
    main()
