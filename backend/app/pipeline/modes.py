"""The six RAG modes. Each returns (passages, facts, notes). Notes describe what the mode did and are
shown to the user, so the behaviour is visible rather than hidden."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..config import get_settings
from .candidates import build_pool, finish
from .options import Options
from .rerank import hub_for
from .techniques import rewrite_queries


@dataclass
class Ctx:
    index: object
    graph: object | None
    ask: object          # callable(prompt: str) -> str
    hub: object


GRADE_PROMPT = (
    "Question: QUESTION\n\nPassage:\nPASSAGE\n\n"
    "Is this passage relevant and useful for answering the question? "
    "Reply with exactly one word: CORRECT, AMBIGUOUS, or INCORRECT."
)
RETRY_PROMPT = (
    "The search for this question found nothing relevant. Write one better search query for finding "
    "the answer in a document. Reply with only the query.\n\nQuestion: QUESTION"
)
AGENT_PROMPT = (
    "Question: QUESTION\n\nPassages gathered so far:\nPREVIEW\n\nQueries already used: USED\n\n"
    "If the passages are enough to answer the question, reply {\"action\": \"answer\"}. "
    "Otherwise reply {\"action\": \"search\", \"query\": \"a new search query\"}. Reply with JSON only."
)


def _parse_grade(reply: str) -> str:
    upper = reply.upper()
    for label in ("INCORRECT", "AMBIGUOUS", "CORRECT"):  # INCORRECT first: it contains CORRECT
        if label in upper:
            return label
    return "INCORRECT"


def grade(ctx: Ctx, question: str, passages: list[dict]) -> list[str]:
    """One LLM call per passage, run concurrently. Returns CORRECT, AMBIGUOUS or INCORRECT for each."""
    def work(p: dict) -> str:
        try:
            prompt = GRADE_PROMPT.replace("QUESTION", question).replace("PASSAGE", p["text"][:2000])
            return _parse_grade(ctx.ask(prompt))
        except Exception:
            return "INCORRECT"

    if not passages:
        return []
    with ThreadPoolExecutor(max_workers=max(1, get_settings().kg_workers)) as pool:
        return list(pool.map(work, passages))


def traditional_passages(ctx: Ctx, question: str, opts: Options) -> tuple[list[dict], list[str]]:
    queries = rewrite_queries(question, ctx.ask) if "query_rewriting" in opts.techniques else [question]
    notes = [f"Query rewriting produced {len(queries)} queries"] if len(queries) > 1 else []
    pool = build_pool(ctx.index, queries, opts)
    return finish(ctx.hub, question, pool, opts), notes


def traditional(ctx: Ctx, question: str, opts: Options):
    passages, notes = traditional_passages(ctx, question, opts)
    return passages, [], notes


def crag(ctx: Ctx, question: str, opts: Options):
    passages, notes = traditional_passages(ctx, question, opts)
    labels = grade(ctx, question, passages)
    keep = [p for p, l in zip(passages, labels) if l == "CORRECT"]
    notes.append(f"CRAG graded {len(keep)} of {len(passages)} passages as correct")
    if keep:
        return keep, [], notes

    retry = ctx.ask(RETRY_PROMPT.replace("QUESTION", question)).strip().strip('"')
    notes.append(f"CRAG found no correct passages; retried with \"{retry}\"")
    passages, more = traditional_passages(ctx, retry or question, opts)
    labels = grade(ctx, question, passages)
    keep = [p for p, l in zip(passages, labels) if l == "CORRECT"]
    notes.extend(more)
    if not keep:
        notes.append("CRAG still found no correct passages; refusing")
    return keep, [], notes


def self_rag_select(ctx: Ctx, question: str, opts: Options):
    """Self-RAG's relevance reflection: keep only passages judged relevant. The answer's support check runs
    after generation, in the answer step."""
    passages, notes = traditional_passages(ctx, question, opts)
    labels = grade(ctx, question, passages)
    keep = [p for p, l in zip(passages, labels) if l == "CORRECT"]
    notes.append(f"Self-RAG kept {len(keep)} of {len(passages)} passages judged relevant")
    return keep, [], notes


def _parse_decision(reply: str) -> dict:
    start, end = reply.find("{"), reply.rfind("}")
    try:
        data = json.loads(reply[start:end + 1]) if start >= 0 and end > start else {}
    except json.JSONDecodeError:
        data = {}
    return data if isinstance(data, dict) else {}


def agentic(ctx: Ctx, question: str, opts: Options):
    # Round zero: search with the question itself, so the model always decides with something in hand.
    gathered, _ = traditional_passages(ctx, question, opts)
    used: list[str] = [question]
    notes: list[str] = [f"Agent's first search added {len(gathered)} passages"]
    for _ in range(get_settings().agentic_max_steps):
        preview = "\n".join(f"[{i + 1}] {p['text'][:200]}" for i, p in enumerate(gathered)) or "(none yet)"
        prompt = (AGENT_PROMPT.replace("QUESTION", question).replace("PREVIEW", preview)
                  .replace("USED", json.dumps(used)))
        decision = _parse_decision(ctx.ask(prompt))
        query = decision.get("query") if decision.get("action") == "search" else None
        if not isinstance(query, str) or not query.strip() or query in used:
            break
        used.append(query)
        new, _ = traditional_passages(ctx, query, opts)
        seen = {p["id"] for p in gathered}
        added = [p for p in new if p["id"] not in seen]
        gathered.extend(added)
        notes.append(f"Agent searched \"{query}\" and added {len(added)} passages")
    if len(used) == 1:
        notes.append("Agent answered without further searches")
    return gathered[:8], [], notes


def lightrag(ctx: Ctx, question: str, opts: Options):
    if ctx.graph is None or ctx.graph.is_empty():
        passages, notes = traditional_passages(ctx, question, opts)
        return passages, [], notes + ["No knowledge graph available; used chunk search only"]
    facts, graph_chunk_ids = ctx.graph.context(question, opts.lightrag_mode)
    queries = rewrite_queries(question, ctx.ask) if "query_rewriting" in opts.techniques else [question]
    pool = build_pool(ctx.index, queries, opts, extra_chunk_ids=graph_chunk_ids)
    return finish(ctx.hub, question, pool, opts), facts, [f"LightRAG {opts.lightrag_mode}: {len(facts)} graph facts"]


def graphrag(ctx: Ctx, question: str, opts: Options):
    if ctx.graph is None or not getattr(ctx.graph, "communities", []):
        passages, notes = traditional_passages(ctx, question, opts)
        return passages, [], notes + ["No graph communities available; used chunk search only"]
    facts, community_chunk_ids = ctx.graph.community_context(question)
    queries = rewrite_queries(question, ctx.ask) if "query_rewriting" in opts.techniques else [question]
    pool = build_pool(ctx.index, queries, opts, extra_chunk_ids=community_chunk_ids)
    return finish(ctx.hub, question, pool, opts), facts, [f"GraphRAG: {len(facts)} community summaries"]


DISPATCH = {"traditional": traditional, "crag": crag, "agentic": agentic, "lightrag": lightrag, "graphrag": graphrag}


def select(ctx: Ctx, question: str, opts: Options):
    """Runs the retrieval part of the chosen mode. Self-RAG's selection is here; its answer check is in rag.py."""
    if opts.mode == "self_rag":
        return self_rag_select(ctx, question, opts)
    return DISPATCH[opts.mode](ctx, question, opts)


def make_ctx(index, graph, ask) -> Ctx:
    return Ctx(index=index, graph=graph, ask=ask, hub=hub_for(index))
