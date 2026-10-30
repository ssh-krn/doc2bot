"""Question answering. The options choose the RAG mode and techniques (app/pipeline). This module wires
them to the chat model, runs Self-RAG's answer check, and streams events.

Events: sources, note (what the pipeline did), facts (graph facts), token, done, error.

VLM-enhanced queries: when vision is enabled and retrieved sources include figures, the figures are
attached to the prompt and answered by the vision model."""

import base64
import time
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from . import images
from .config import get_settings
from .llm import get_llm
from .pipeline.catalog import hybrid_label
from .pipeline.modes import make_ctx, select
from .pipeline.options import MODES, Options, make_options
from .retrieval import Index

SYSTEM_PROMPT = (
    "You answer questions using only the material provided. Claims must be supported by the "
    "numbered sources, and you must cite them inline, like [1], right after each claim. "
    "Knowledge graph facts and community summaries are background, not citable, so cite the "
    "numbered sources they come from. Attached figures are sources too: read them carefully, "
    "including any values or labels. If the material does not contain the answer, say you couldn't "
    "find it in the documents. Do not use outside knowledge."
)
REFUSAL = "I couldn't find anything relevant in your documents to answer that."
SELF_RAG_REGENERATE = (
    "\n\nYour previous draft contained claims the sources do not support. Rewrite the answer, keeping "
    "only claims that the numbered sources directly support, with citations."
)
JUDGE_PROMPT = (
    "Sources:\nSOURCES\n\nAnswer:\nANSWER\n\n"
    "Is every claim in the answer supported by the sources? "
    "Reply with exactly one word: SUPPORTED, PARTIAL, or UNSUPPORTED."
)


def _text_of(chunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def _ask_text(prompt: str) -> str:
    """Text-only call to the chat model. Looked up at call time so it can be replaced in tests."""
    return _text_of(get_llm().invoke([HumanMessage(content=prompt)]))


def _sources(passages: list[dict]) -> list[dict]:
    return [
        {"index": i, "filename": p["filename"], "page": p["page"], "text": p["text"],
         "score": p["score"], "modality": p.get("modality", "text"), "image_key": p.get("image_key") or ""}
        for i, p in enumerate(passages, start=1)
    ]


def _select(index: Index, graph, question: str, opts: Options):
    """Returns (sources, facts, notes). Graph modes fall back to chunk search when no graph exists."""
    notes = []
    if opts.mode in ("lightrag", "graphrag") and graph is None:
        notes.append(f"Knowledge graph is disabled; {opts.mode} ran as chunk search")
    ctx = make_ctx(index, graph, _ask_text)
    passages, facts, mode_notes = select(ctx, question, opts)
    return _sources(passages), facts, notes + mode_notes


def retrieve(index: Index, graph, question: str, mode: str = "traditional", opts: Options | None = None):
    """Returns (numbered sources, graph facts). Kept for the evaluation harness and tests."""
    opts = opts or make_options(mode)
    sources, facts, _ = _select(index, graph, question, opts)
    return sources, facts


def _figure_parts(sources: list[dict], enabled: bool) -> list[dict]:
    if not enabled:
        return []
    limit = get_settings().vlm_query_max_images
    parts = []
    for s in sources:
        if len(parts) >= limit:
            break
        if not s.get("image_key"):
            continue
        loaded = images.load(s["image_key"])
        if loaded is None:
            continue
        data, mime = loaded
        b64 = base64.b64encode(data).decode("ascii")
        parts.append({"type": "image_url", "image_url": f"data:{mime};base64,{b64}"})
    return parts


def _text_block(question: str, sources: list[dict], facts: list[str]) -> str:
    parts = []
    if facts:
        bullets = "\n".join(f"- {f}" for f in facts[:24])
        parts.append(f"Background (not citable):\n{bullets}")
    context = "\n\n".join(
        f"[{s['index']}] ({s['filename']}, page {s['page']}, {s['modality']})\n{s['text']}" for s in sources
    )
    parts.append(f"Sources:\n{context}")
    return "\n\n".join(parts) + f"\n\nQuestion: {question}"


def _plan(question: str, sources: list[dict], facts: list[str], vision: bool | None, extra: str = ""):
    """Returns (messages, llm). Uses the vision model only when figures are attached."""
    use_vision = get_settings().enable_vlm and (True if vision is None else vision)
    figures = _figure_parts(sources, use_vision)
    text = _text_block(question, sources, facts) + extra
    if figures:
        from .parsing.vision import vision_llm  # imported lazily: needs a vision provider key
        content = [{"type": "text", "text": text}, *figures]
        llm = vision_llm()
    else:
        content = text
        llm = get_llm()
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)], llm


def _self_rag_answer(question: str, sources: list[dict], facts: list[str], vision: bool | None):
    """Generate, check the answer against its sources, and regenerate once if it isn't fully supported.
    Returns (answer, notes)."""
    messages, llm = _plan(question, sources, facts, vision)
    answer = _text_of(llm.invoke(messages))
    notes = []
    verdict = _judge(answer, sources)
    notes.append(f"Self-RAG support check: {verdict}")
    if verdict != "SUPPORTED":
        messages, llm = _plan(question, sources, facts, vision, extra=SELF_RAG_REGENERATE)
        answer = _text_of(llm.invoke(messages))
        verdict = _judge(answer, sources)
        notes.append(f"Self-RAG regenerated; support check: {verdict}")
    return answer, notes


def _judge(answer: str, sources: list[dict]) -> str:
    source_text = "\n".join(f"[{s['index']}] {s['text']}" for s in sources)
    reply = _ask_text(JUDGE_PROMPT.replace("SOURCES", source_text).replace("ANSWER", answer)).upper()
    for label in ("UNSUPPORTED", "PARTIAL", "SUPPORTED"):  # UNSUPPORTED first: it contains SUPPORTED
        if label in reply:
            return label
    return "UNSUPPORTED"


def _options(mode, lightrag_mode, techniques, rerankers, vision) -> Options:
    return make_options(mode=mode, lightrag_mode=lightrag_mode or "hybrid", techniques=techniques,
                        rerankers=rerankers, vision=vision)


def answer_stream(index: Index, graph, question: str, mode: str = "traditional", vision: bool | None = None,
                  techniques=None, rerankers=None, lightrag_mode: str | None = None) -> Iterator[dict]:
    opts = _options(mode, lightrag_mode, techniques, rerankers, vision)
    sources, facts, notes = _select(index, graph, question, opts)
    yield {"type": "sources", "data": sources}
    for note in notes:
        yield {"type": "note", "data": note}
    if facts:
        yield {"type": "facts", "data": facts}
    if not sources:
        yield {"type": "token", "data": REFUSAL}
        yield {"type": "done"}
        return

    if opts.mode == "self_rag":
        answer, check_notes = _self_rag_answer(question, sources, facts, vision)
        for note in check_notes:
            yield {"type": "note", "data": note}
        yield {"type": "token", "data": answer}
        yield {"type": "done"}
        return

    messages, llm = _plan(question, sources, facts, vision)
    for chunk in llm.stream(messages):
        text = _text_of(chunk)
        if text:
            yield {"type": "token", "data": text}
    yield {"type": "done"}


def answer_once(index: Index, graph, question: str, mode: str = "traditional", vision: bool | None = None,
                techniques=None, rerankers=None, lightrag_mode: str | None = None) -> dict:
    """Non-streaming answer with timings and token counts, used by the evaluation."""
    opts = _options(mode, lightrag_mode, techniques, rerankers, vision)
    t0 = time.perf_counter()
    sources, facts, notes = _select(index, graph, question, opts)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    base = {"sources": sources, "facts": facts, "notes": notes, "retrieval_ms": retrieval_ms,
            "hybrid_label": hybrid_label(opts)}
    if not sources:
        return {**base, "answer": REFUSAL, "input_tokens": 0, "output_tokens": 0, "llm_ms": 0.0}
    t1 = time.perf_counter()
    if opts.mode == "self_rag":
        answer, check_notes = _self_rag_answer(question, sources, facts, vision)
        return {**base, "answer": answer, "notes": notes + check_notes,
                "llm_ms": (time.perf_counter() - t1) * 1000, "input_tokens": 0, "output_tokens": 0}
    messages, llm = _plan(question, sources, facts, vision)
    resp = llm.invoke(messages)
    usage = getattr(resp, "usage_metadata", None) or {}
    return {**base, "answer": _text_of(resp), "llm_ms": (time.perf_counter() - t1) * 1000,
            "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}


__all__ = ["retrieve", "answer_stream", "answer_once", "MODES", "REFUSAL"]
