import re
import json
from pathlib import Path


def norm(s: str) -> str:
    """Normalize text so snippets match despite PDF line breaks, hyphenation, and case."""
    s = re.sub(r"-\s*\n\s*", "", s)
    return re.sub(r"\s+", " ", s).lower().strip()


def load_qa(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def is_gold_chunk(chunk: dict, qa: dict) -> bool:
    """A retrieved chunk counts as relevant if it comes from the gold page and contains the gold snippet.
    Matching on the snippet (not chunk id) keeps labels valid across different chunk sizes."""
    return (
        chunk["filename"] == qa["gold_doc"]
        and chunk["page"] == qa["gold_page"]
        and norm(qa["gold_snippet"]) in norm(chunk["text"])
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    k = max(0, min(len(vals) - 1, round(p / 100 * (len(vals) - 1))))
    return vals[k]
