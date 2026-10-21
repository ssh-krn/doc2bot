import time
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_settings
from .llm import get_llm
from .retrieval import Index

SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources provided. "
    "Cite the sources you use inline, like [1], right after each claim. "
    "If the sources do not contain the answer, say you couldn't find it in the documents. "
    "Do not use outside knowledge."
)
REFUSAL = "I couldn't find anything relevant in your documents to answer that."


def retrieve(index: Index, question: str) -> list[dict]:
    s = get_settings()
    passages = index.search(question, mode="hybrid", rerank=True, k=s.rerank_top_n, candidates=s.dense_k)
    passages = [p for p in passages if p["score"] >= s.rerank_min_score]
    return [
        {"index": i, "filename": p["filename"], "page": p["page"], "text": p["text"], "score": p["score"]}
        for i, p in enumerate(passages, start=1)
    ]


def _messages(question: str, sources: list[dict]) -> list:
    context = "\n\n".join(
        f"[{s['index']}] ({s['filename']}, page {s['page']})\n{s['text']}" for s in sources
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Sources:\n{context}\n\nQuestion: {question}"),
    ]


def _text_of(chunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def answer_stream(index: Index, question: str) -> Iterator[dict]:
    """Streaming events: one sources event, then token events, then done."""
    sources = retrieve(index, question)
    yield {"type": "sources", "data": sources}
    if not sources:
        yield {"type": "token", "data": REFUSAL}
        yield {"type": "done"}
        return
    for chunk in get_llm().stream(_messages(question, sources)):
        text = _text_of(chunk)
        if text:
            yield {"type": "token", "data": text}
    yield {"type": "done"}


def answer_once(index: Index, question: str) -> dict:
    """Non-streaming answer with timings and token counts, used by the evaluation."""
    t0 = time.perf_counter()
    sources = retrieve(index, question)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    if not sources:
        return {"answer": REFUSAL, "sources": [], "input_tokens": 0, "output_tokens": 0,
                "retrieval_ms": retrieval_ms, "llm_ms": 0.0}
    t1 = time.perf_counter()
    resp = get_llm().invoke(_messages(question, sources))
    llm_ms = (time.perf_counter() - t1) * 1000
    usage = getattr(resp, "usage_metadata", None) or {}
    return {
        "answer": _text_of(resp),
        "sources": sources,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
    }
