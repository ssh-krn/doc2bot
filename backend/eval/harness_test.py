"""Unit tests for the evaluation harness: label matching, multi-passage recall, failure classes, summaries and
report formats. No models or API keys needed. Run from backend/: python -m eval.harness_test"""

import json
import tempfile
from pathlib import Path

from eval.common import all_golds_found, load_qa, matches, norm
from eval.run_eval import (ablation_markdown, classify_failure, compare_markdown, run_retrieval_fn, summarize)

DOC_A, DOC_B = "a.pdf", "b.pdf"
GOLD_A = {"doc": DOC_A, "page": 1, "snippet": "The   refund window\nis 30 days."}
GOLD_B = {"doc": DOC_B, "page": 2, "snippet": "Shipping takes 5 days."}


def chunk(doc, page, text):
    return {"filename": doc, "page": page, "text": text}


def main():
    # normalization: whitespace, line breaks and case
    assert norm("The   refund window\nis 30 days.") == "the refund window is 30 days."
    assert matches(chunk(DOC_A, 1, "...the refund window is 30 days..."), GOLD_A)
    assert not matches(chunk(DOC_A, 2, "...the refund window is 30 days..."), GOLD_A), "wrong page must not match"
    assert not matches(chunk(DOC_B, 1, "...the refund window is 30 days..."), GOLD_A), "wrong doc must not match"

    # multi-hop: both passages must be present
    qa = {"golds": [GOLD_A, GOLD_B]}
    assert not all_golds_found([chunk(DOC_A, 1, "the refund window is 30 days.")], qa)
    assert all_golds_found([chunk(DOC_A, 1, "the refund window is 30 days."),
                            chunk(DOC_B, 2, "Shipping takes 5 days.")], qa)

    # loading: old single-passage fields become golds; type defaults follow answerability
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qa.jsonl"
        path.write_text("\n".join([
            json.dumps({"id": "old", "question": "q", "gold_doc": DOC_A, "gold_page": 1,
                        "gold_snippet": "x", "gold_answer_keywords": ["x"]}),
            json.dumps({"id": "none", "question": "q", "answerable": False}),
        ]), encoding="utf-8")
        items = load_qa(str(path))
        assert items[0]["golds"] == [{"doc": DOC_A, "page": 1, "snippet": "x"}] and items[0]["type"] == "fact"
        assert items[1]["type"] == "unanswerable" and items[1]["golds"] == []

    # retrieval runner: recall, MRR and per-type breakdown
    items = [{"id": "1", "type": "fact", "question": "a", "answerable": True, "golds": [GOLD_A]},
             {"id": "2", "type": "multi_hop", "question": "b", "answerable": True, "golds": [GOLD_A, GOLD_B]}]
    answers = {"a": [chunk(DOC_A, 1, "the refund window is 30 days.")],
               "b": [chunk(DOC_A, 1, "the refund window is 30 days."), chunk(DOC_B, 2, "Shipping takes 5 days.")]}
    m = run_retrieval_fn(lambda q: answers[q], items, ks=(1, 3))
    assert m["recall@1"] == 0.5 and m["recall@3"] == 1.0 and m["mrr@5"] == 1.0  # multi-hop needs both passages, so k=1 can miss it
    assert m["by_type"] == {"fact": 1.0, "multi_hop": 1.0}
    m = run_retrieval_fn(lambda q: [chunk(DOC_A, 1, "the refund window is 30 days.")] if q == "b" else answers[q],
                         items, ks=(1,))
    assert m["by_type"]["multi_hop"] == 0.0, "multi-hop with a missing hop must fail"

    # failure classes
    base = {"answerable": True, "correct": True, "refused": False, "retrieved_gold": True, "bad_cites": 0}
    assert classify_failure(base, True) is None
    assert classify_failure({**base, "retrieved_gold": False}, True) == "retrieval_miss"
    assert classify_failure({**base, "retrieved_gold": False}, False) == "snippet_split_by_chunking"
    assert classify_failure({**base, "refused": True, "correct": False}, True) == "false_refusal"
    assert classify_failure({**base, "correct": False}, True) == "wrong_answer"
    assert classify_failure({**base, "bad_cites": 1}, True) == "bad_citation"
    assert classify_failure({"answerable": False, "refused": False, "correct": False}, True) == "hallucinated_answer"
    assert classify_failure({"answerable": False, "refused": True, "correct": True}, True) is None

    # summary: accuracy by type and refusal accuracy
    results = [
        {"answerable": True, "type": "fact", "correct": True, "bad_cites": 0, "n_cites": 2,
         "retrieval_ms": 1, "llm_ms": 2, "input_tokens": 100, "output_tokens": 10},
        {"answerable": True, "type": "multi_hop", "correct": False, "bad_cites": 1, "n_cites": 2,
         "retrieval_ms": 1, "llm_ms": 2, "input_tokens": 120, "output_tokens": 12},
        {"answerable": False, "type": "unanswerable", "correct": True, "bad_cites": 0, "n_cites": 0,
         "retrieval_ms": 1, "llm_ms": 0, "input_tokens": 0, "output_tokens": 0},
    ]
    s = summarize(results, 800, "crag")
    assert s["answer_accuracy_by_type"] == {"fact": 1.0, "multi_hop": 0.0}
    assert s["refusal_accuracy_unanswerable"] == 1.0
    assert abs(s["citation_error_rate"] - 0.25) < 1e-9

    # report formats render
    rows = [{"chunk_size": 800, "mode": "bm25", "rerank": False, "label_coverage": 1.0, "recall@1": 0.5,
             "recall@3": 0.7, "recall@5": 0.8, "mrr@5": 0.6, "p50_ms": 0.1, "p95_ms": 0.2,
             "by_type": {"fact": 0.9}}]
    md = ablation_markdown(rows)
    assert "| 800 | bm25 | no | 50% |" in md and "Recall@5 by question type" in md
    assert "| crag |" in compare_markdown([s])
    print("HARNESS TESTS PASSED")


if __name__ == "__main__":
    main()
