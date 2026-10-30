"""Checks eval/qa.jsonl against the rendered corpus. Every gold snippet must appear on its stated page, and every
answer keyword must appear in the gold passages. Exits non-zero on any failure, so a bad label can't pass silently.

Run from backend/: python -m eval.validate_qa
"""

import collections
import json
import sys
from pathlib import Path

import pymupdf

from eval.common import norm

QA = Path("eval/qa.jsonl")
CORPUS = Path("eval/corpus")
VALID_TYPES = {"fact", "definition", "comparison", "numeric", "multi_hop", "unanswerable"}


def page_texts() -> dict[tuple[str, int], str]:
    pages = {}
    for pdf in sorted(CORPUS.glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            for n, page in enumerate(doc, start=1):
                pages[(pdf.name, n)] = norm(page.get_text())
    return pages


def main() -> int:
    if not QA.exists() or not CORPUS.exists():
        print("Build the corpus and questions first: python -m eval.build_corpus && python -m eval.build_qa")
        return 1
    pages = page_texts()
    items = [json.loads(line) for line in QA.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures: list[str] = []
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        failures.append("duplicate question ids")

    for it in items:
        qid, qtype = it["id"], it["type"]
        if qtype not in VALID_TYPES:
            failures.append(f"{qid}: unknown type {qtype}")
        if qtype == "unanswerable":
            if it["golds"] or it["answerable"]:
                failures.append(f"{qid}: unanswerable question must have no golds and answerable=false")
            continue
        if not it["answerable"] or not it["golds"]:
            failures.append(f"{qid}: answerable question needs golds")
            continue
        for gold in it["golds"]:
            text = pages.get((gold["doc"], gold["page"]))
            if text is None:
                failures.append(f"{qid}: page {gold['doc']} p{gold['page']} does not exist")
            elif norm(gold["snippet"]) not in text:
                failures.append(f"{qid}: snippet not on {gold['doc']} p{gold['page']}: {gold['snippet'][:60]!r}")
        if qtype == "multi_hop" and len({g["doc"] for g in it["golds"]}) < 2:
            failures.append(f"{qid}: multi_hop needs passages from at least two documents")
        gold_text = " ".join(pages.get((g["doc"], g["page"]), "") for g in it["golds"])
        for kw in it["gold_answer_keywords"]:
            if norm(kw) not in gold_text:
                failures.append(f"{qid}: keyword {kw!r} not in its gold passages")

    counts = collections.Counter(it["type"] for it in items)
    print(f"{len(items)} questions:", dict(counts))
    if failures:
        print(f"{len(failures)} problem(s):")
        for f in failures:
            print("  -", f)
        return 1
    print("all gold snippets, pages and keywords check out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
