"""Retrieval techniques that act on candidate passages. Each one is a plain function over passage dicts
(id, doc_id, filename, page, text, parent_id, parent_text, ...), so they can be tested without a model."""

import json
import re
from collections import Counter

from .rerank import RerankerHub

REWRITE_PROMPT = (
    "Rewrite the question as 3 different search queries that could find the answer in a document. "
    "Vary the wording and the keywords. Return only a JSON array of strings, with no other text.\n\n"
    "Question: QUESTION"
)


def rewrite_queries(question: str, ask) -> list[str]:
    """The original question first, then up to three LLM rewrites. Falls back to the question alone."""
    try:
        raw = ask(REWRITE_PROMPT.replace("QUESTION", question))
        start, end = raw.find("["), raw.rfind("]")
        items = json.loads(raw[start:end + 1]) if start >= 0 and end > start else []
    except Exception:
        items = []
    queries = [question]
    for item in items:
        if isinstance(item, str) and item.strip() and item.strip() not in queries:
            queries.append(item.strip())
    return queries[:4]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def compress(passages: list[dict], question: str, hub: RerankerHub, names: tuple, keep: int = 2) -> list[dict]:
    """Contextual compression: keeps the `keep` sentences of each passage that the reranker scores highest,
    in their original order."""
    out = []
    for p in passages:
        sentences = split_sentences(p["text"])
        if len(sentences) <= keep:
            out.append(p)
            continue
        ranked = hub.rank(names, question, [{"text": s} for s in sentences])
        best = {sentences.index(c["text"]) for c, _, _ in ranked[:keep]}
        text = " ".join(s for i, s in enumerate(sentences) if i in best)
        out.append({**p, "text": text, "compressed": True})
    return out


def sentence_window(passages: list[dict], question: str, hub: RerankerHub, names: tuple, radius: int = 2) -> list[dict]:
    """Sentence window retrieval: finds the best sentence in each passage and returns it with `radius`
    sentences of context on each side."""
    out = []
    for p in passages:
        sentences = split_sentences(p["text"])
        if len(sentences) <= 1:
            out.append(p)
            continue
        ranked = hub.rank(names, question, [{"text": s} for s in sentences])
        best = sentences.index(ranked[0][0]["text"])
        window = sentences[max(0, best - radius): best + radius + 1]
        out.append({**p, "text": " ".join(window), "windowed": True})
    return out


def parent_expand(passages: list[dict]) -> list[dict]:
    """Parent document retrieval: each child is replaced by its parent passage. Duplicate parents are dropped."""
    seen, out = set(), []
    for p in passages:
        key = (p.get("doc_id"), p.get("parent_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append({**p, "text": p.get("parent_text") or p["text"], "expanded": "parent"})
    return out


def auto_merge(passages: list[dict], min_children: int = 2) -> list[dict]:
    """Auto-merging retrieval: where at least `min_children` retrieved children share a parent, one passage
    holding the parent replaces them. A lone child stays as it is."""
    counts = Counter((p.get("doc_id"), p.get("parent_id")) for p in passages)
    out, emitted = [], set()
    for p in passages:
        key = (p.get("doc_id"), p.get("parent_id"))
        if counts[key] >= min_children:
            if key not in emitted:
                emitted.add(key)
                out.append({**p, "text": p.get("parent_text") or p["text"], "expanded": "merged"})
        else:
            out.append(p)
    return out
