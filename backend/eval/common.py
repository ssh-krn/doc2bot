import json
import re
from pathlib import Path


def norm(s: str) -> str:
    """Normalize text so snippets match despite PDF line breaks, hyphenation, and case."""
    s = re.sub(r"-\s*\n\s*", "", s)
    return re.sub(r"\s+", " ", s).lower().strip()


def load_qa(path: str) -> list[dict]:
    """Loads questions. Each item gets `golds` (a list of passages). The older single-passage fields
    (gold_doc, gold_page, gold_snippet) are still accepted, and an item without a `type` gets one from answerable."""
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if "golds" not in it:
            it["golds"] = (
                [{"doc": it["gold_doc"], "page": it["gold_page"], "snippet": it["gold_snippet"]}]
                if it.get("gold_snippet") else []
            )
        it.setdefault("answerable", bool(it["golds"]))
        it.setdefault("type", "fact" if it["answerable"] else "unanswerable")
        it.setdefault("gold_answer_keywords", [])
        items.append(it)
    return items


def matches(chunk: dict, gold: dict) -> bool:
    """A retrieved chunk supports a gold passage if it comes from the gold page and contains the gold snippet.
    Matching on text rather than chunk id keeps labels valid across chunk sizes."""
    return (
        chunk.get("filename") == gold["doc"]
        and chunk.get("page") == gold["page"]
        and norm(gold["snippet"]) in norm(chunk.get("text", ""))
    )


def is_gold_chunk(chunk: dict, qa: dict) -> bool:
    return any(matches(chunk, g) for g in qa["golds"])


def all_golds_found(chunks: list[dict], qa: dict) -> bool:
    return bool(qa["golds"]) and all(any(matches(c, g) for c in chunks) for g in qa["golds"])


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    k = max(0, min(len(vals) - 1, round(p / 100 * (len(vals) - 1))))
    return vals[k]
