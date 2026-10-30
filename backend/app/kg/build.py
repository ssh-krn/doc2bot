"""Builds the graph for a document: one extraction call per chunk, run with bounded concurrency
and retried on failure. A chunk that still fails is skipped and logged, so one bad response
doesn't fail the upload."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from .extract import build_prompt, parse_extraction
from .graph import KnowledgeGraph

log = logging.getLogger("kg")


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def llm_invoke_factory():
    from langchain_core.messages import HumanMessage  # local import: keeps this module light for tests

    from ..llm import get_llm

    llm = get_llm()

    def invoke(prompt: str) -> str:
        return _text_of(llm.invoke([HumanMessage(content=prompt)]).content)

    return invoke


def extract_chunk(invoke, text: str, retries: int = 3) -> tuple[list[dict], list[dict]]:
    for attempt in range(retries):
        try:
            return parse_extraction(invoke(build_prompt(text)))
        except Exception as e:  # rate limits, timeouts, provider errors
            if attempt == retries - 1:
                log.warning("extraction failed after %d attempts: %s", retries, e)
                return [], []
            time.sleep(2 ** attempt)
    return [], []


def index_document(graph: KnowledgeGraph, invoke, doc_id: str, chunks: list[dict], workers: int = 2) -> dict:
    """chunks: dicts with 'id' and 'text'. Returns counts for this document."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda c: (c["id"], *extract_chunk(invoke, c["text"])), chunks))
    entities = relations = 0
    for chunk_id, ents, rels in results:
        graph.add_extraction(doc_id, chunk_id, ents, rels)
        entities += len(ents)
        relations += len(rels)
    graph.save()
    return {"entities_extracted": entities, "relations_extracted": relations}


SUMMARY_PROMPT = """Below are entities and relations extracted from a document collection.
Write a 3-5 sentence summary of the main topic and how these items relate. Use only the facts given.

Facts:
<<<
FACTS
>>>"""


def llm_summarize_factory():
    """Returns summarize(list_of_fact_strings) -> summary text, used for GraphRAG communities."""
    from langchain_core.messages import HumanMessage  # local import, as in llm_invoke_factory

    from ..llm import get_llm

    llm = get_llm()

    def summarize(facts: list[str]) -> str:
        prompt = SUMMARY_PROMPT.replace("FACTS", "\n".join(f"- {f}" for f in facts))
        return _text_of(llm.invoke([HumanMessage(content=prompt)]).content)

    return summarize
