"""Offline tests for RAG modes, retrieval techniques, and rerankers. All models are stubbed: the chat model
answers by prompt type, the embedder is a hashing stub, and rerankers are fake scorers. Tests check the
pipeline logic (routing, expansion, gating, fusion, error messages), not retrieval or answer quality.
Run from backend/: python -m eval.pipeline_smoke_test"""

import json
import os
import sys
import tempfile
import types
import uuid
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["IMAGE_DIR"] = str(Path(_tmp) / "images")
os.environ["KG_PATH"] = str(Path(_tmp) / "kg.json")
os.environ["QDRANT_PATH"] = str(Path(_tmp) / "qdrant")
os.environ["ENABLE_VLM"] = "false"
os.environ["COHERE_API_KEY"] = ""

from langchain_core.messages import AIMessage  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

import app.kg.build as kg_build  # noqa: E402
import app.rag as rag  # noqa: E402
from app.ingest import process_document  # noqa: E402
from app.kg.build import index_document  # noqa: E402
from app.kg.graph import KnowledgeGraph  # noqa: E402
from app.parsing.types import ParseError  # noqa: E402
from app.pipeline.catalog import hybrid_label  # noqa: E402
from app.pipeline.options import Options, make_options  # noqa: E402
from app.pipeline.rerank import RerankerHub  # noqa: E402
from app.pipeline.techniques import auto_merge, compress, parent_expand, sentence_window, split_sentences  # noqa: E402
from app.retrieval import Index  # noqa: E402
from eval.kg_smoke_test import pdf_bytes, stub_llm  # noqa: E402
from eval.smoke_test import DIM, StubEmbedder  # noqa: E402


class FakeChat:
    """Answers each prompt type the way a model would, with knobs for the judgement steps."""

    def __init__(self):
        self.grade = "CORRECT"
        self.judge_replies = ["SUPPORTED"]
        self.decisions = ['{"action": "answer"}']
        self.calls = {"rewrite": 0, "grade": 0, "judge": 0, "decide": 0, "answer": 0, "summary": 0}

    def _reply(self, text: str) -> str:
        if "3 different search queries" in text:
            self.calls["rewrite"] += 1
            return '["acme headquarters", "jane doe employer", "where acme operates"]'
        if "exactly one word: CORRECT" in text:
            self.calls["grade"] += 1
            return self.grade
        if "exactly one word: SUPPORTED" in text:
            self.calls["judge"] += 1
            return self.judge_replies.pop(0) if len(self.judge_replies) > 1 else self.judge_replies[0]
        if "Reply with JSON only" in text:
            self.calls["decide"] += 1
            return self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
        if "Reply with only the query" in text:
            return "acme berlin office"
        if "3-5 sentence summary" in text:
            self.calls["summary"] += 1
            return "Jane Doe works at Acme Corp, which is headquartered in Berlin."
        self.calls["answer"] += 1
        return "Jane Doe is chief scientist at Acme Corp [1]."

    def invoke(self, messages):
        content = messages[-1].content
        text = content if isinstance(content, str) else "".join(p.get("text", "") for p in content)
        return AIMessage(content=self._reply(text),
                         usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60})

    def stream(self, messages):
        yield AIMessage(content=self.invoke(messages).content)


class FakeCross:
    def predict(self, pairs):
        return [float(len(set(q.lower().split()) & set(t.lower().split()))) for q, t in pairs]


def build_world():
    index = Index(StubEmbedder(), QdrantClient(":memory:"), "pipe", DIM, reranker=None)
    graph = KnowledgeGraph(StubEmbedder(), path=None)
    docs = {
        "people.pdf": pdf_bytes(["Jane Doe is chief scientist at Acme Corp.",
                                 "Acme Corp is headquartered in Berlin. The office has a lab.",
                                 "The lab runs weekly reviews. Reviews are open to staff."]),
        "events.pdf": pdf_bytes(["Globex organizes the Data Summit in Berlin."]),
    }
    for name, data in docs.items():
        chunks, _ = process_document(name, data, 800, 120)
        doc_id = str(uuid.uuid4())
        ids = index.add_document(doc_id, name, chunks)
        index_document(graph, stub_llm, doc_id, [{"id": i, "text": c["text"]} for i, c in zip(ids, chunks)], workers=2)
    graph.refresh_communities(summarize=lambda facts: "Jane Doe works at Acme Corp in Berlin.")
    return index, graph


def main():
    chat = FakeChat()
    rag._ask_text.__globals__["get_llm"] = lambda: chat  # rag._ask_text looks get_llm up at call time
    rag.get_llm = lambda: chat
    kg_build.get_llm = lambda: chat
    index, graph = build_world()
    Q = "Who is Jane Doe and where is her employer based?"

    # 1. Options validation and aliases
    assert make_options("naive").mode == "traditional"
    assert make_options("local").lightrag_mode == "local" and make_options("local").mode == "lightrag"
    assert make_options("traditional").rerankers == ("cross_encoder",)
    assert make_options("traditional").techniques == frozenset({"hybrid"})
    for bad in [dict(mode="magic"), dict(mode="traditional", techniques=["nope"]),
                dict(mode="traditional", rerankers=["gpt"]), dict(mode="lightrag", lightrag_mode="deep")]:
        try:
            make_options(**bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    print("options OK")

    # 2. Hybrid names state the RAG type and the search types
    assert hybrid_label(make_options("traditional")) == "Hybrid Traditional RAG · BM25 keyword + dense vector"
    assert hybrid_label(make_options("crag", techniques=[])) == "Dense-only CRAG (Corrective RAG) · dense vector search"
    assert "(local)" in hybrid_label(make_options("lightrag", lightrag_mode="local"))
    print("hybrid labels OK:", hybrid_label(make_options("graphrag")))

    # 3. Every mode runs end to end and returns a cited answer
    for mode in ["traditional", "crag", "self_rag", "lightrag", "agentic", "graphrag"]:
        out = rag.answer_once(index, graph, Q, mode)
        assert out["sources"] and "[1]" in out["answer"], (mode, out["answer"])
        print(f"mode {mode:12s} sources={len(out['sources'])} facts={len(out['facts'])} notes={out['notes'][:1]}")
    assert graph.communities, "GraphRAG needs at least one community"
    assert rag.answer_once(index, graph, Q, "graphrag")["facts"][0].startswith("Community summary")

    # 4. LightRAG sub-modes produce graph facts
    for sub in ["local", "global", "hybrid"]:
        assert rag.answer_once(index, graph, Q, "lightrag", lightrag_mode=sub)["facts"], sub
    print("lightrag sub-modes OK")

    # 5. Graph modes fall back to chunk search when no graph exists
    out = rag.answer_once(index, None, Q, "graphrag")
    assert out["sources"] and any("disabled" in n or "No graph" in n for n in out["notes"]), out["notes"]

    # 6. Techniques
    calls = []
    real_search = index.search
    def spy(query, mode="hybrid", **kw):
        calls.append(mode)
        return real_search(query, mode=mode, **kw)
    index.search = spy
    rag.retrieve(index, None, Q, "traditional", make_options("traditional", techniques=["hybrid"]))
    assert set(calls) == {"hybrid"}, calls
    calls.clear()
    rag.retrieve(index, None, Q, "traditional", make_options("traditional", techniques=[]))
    assert set(calls) == {"dense"}, calls
    index.search = real_search

    chat.calls["rewrite"] = 0
    out = rag.answer_once(index, None, Q, "traditional", techniques=["query_rewriting"])
    assert chat.calls["rewrite"] == 1 and any("produced 4 queries" in n for n in out["notes"]), out["notes"]

    base = rag.retrieve(index, None, Q, "traditional")[0]
    parent_src = rag.retrieve(index, None, Q, "traditional", make_options("traditional", techniques=["parent_document"]))[0]
    parent_texts = {c["parent_text"] for c in index.chunks}
    assert all(s["text"] in parent_texts for s in parent_src), "parent_document must return stored parent passages"

    lab = [{"id": "a", "doc_id": "d", "parent_id": "p1", "parent_text": "PARENT FULL", "text": "child a", "page": 1},
           {"id": "b", "doc_id": "d", "parent_id": "p1", "parent_text": "PARENT FULL", "text": "child b", "page": 1},
           {"id": "c", "doc_id": "d", "parent_id": "p2", "parent_text": "OTHER", "text": "child c", "page": 2}]
    merged = auto_merge(lab)
    assert [m["text"] for m in merged] == ["PARENT FULL", "child c"], merged
    assert len(parent_expand(lab)) == 2
    print("auto-merging and parent expansion OK")

    hub = RerankerHub(StubEmbedder(), FakeCross())
    sents = [{"text": "Jane leads the lab."}, {"text": "The weather was mild."},
             {"text": "The lab reports to Acme."}, {"text": "Coffee is served daily."}]
    comp = compress([{"text": " ".join(s["text"] for s in sents), "id": "x", "doc_id": "d", "parent_id": "p"}],
                    "Who leads the lab?", hub, ("cross_encoder",), keep=2)[0]
    assert comp["compressed"] and len(split_sentences(comp["text"])) == 2
    assert comp["text"].startswith("Jane") or "lab" in comp["text"]
    win = sentence_window([{"text": " ".join(s["text"] for s in sents), "id": "x"}], "Who leads the lab?", hub, ("cross_encoder",), radius=1)[0]
    assert win["windowed"] and len(split_sentences(win["text"])) <= 3
    print("compression and sentence window OK")

    # 7. Rerankers
    names = ("cross_encoder",)
    chunks = [{"id": "1", "text": "Jane leads the lab"}, {"id": "2", "text": "unrelated text"}]
    ranked = hub.rank(names, "who leads the lab", chunks)
    assert ranked[0][0]["id"] == "1" and ranked[0][1] is not None
    ranked = RerankerHub(StubEmbedder(), None).rank(("bi_encoder",), "jane lab", chunks)
    assert ranked[0][1] is None, "bi-encoder scores must not gate refusal"
    fused = hub.rank(("cross_encoder", "bi_encoder"), "who leads the lab", chunks)
    assert len(fused) == 2 and fused[0][1] is not None
    print("bi-encoder and fused rerankers OK")

    fake_fr = types.ModuleType("flashrank")
    class Ranker:
        def rerank(self, req):
            return [{"id": p["id"], "score": 1.0 - 0.1 * p["id"]} for p in req["passages"]] if isinstance(req, dict) else \
                [{"id": p["id"], "score": 1.0 - 0.1 * p["id"]} for p in req.passages]
    class RerankRequest:
        def __init__(self, query, passages):
            self.query, self.passages = query, passages
    fake_fr.Ranker, fake_fr.RerankRequest = Ranker, RerankRequest
    sys.modules["flashrank"] = fake_fr
    fr_scores = hub.score("flashrank", "q", chunks)
    assert fr_scores == [1.0, 0.9], fr_scores
    del sys.modules["flashrank"]

    try:
        hub.score("cohere", "q", chunks)
        raise AssertionError("cohere without a key must fail clearly")
    except ParseError as e:
        assert "COHERE_API_KEY" in str(e)
    print("flashrank mapping and missing-key errors OK (library APIs unverified)")

    # 8. CRAG: correction retry, then refusal when nothing is relevant
    chat.grade = "CORRECT"
    out = rag.answer_once(index, None, Q, "crag")
    assert out["sources"] and "[1]" in out["answer"]
    chat.grade = "INCORRECT"
    out = rag.answer_once(index, None, Q, "crag")
    assert out["answer"] == rag.REFUSAL and any("refusing" in n for n in out["notes"]), out["notes"]
    chat.grade = "CORRECT"
    print("CRAG correction and refusal OK")

    # 9. Self-RAG: an unsupported draft is regenerated
    chat.judge_replies = ["UNSUPPORTED", "SUPPORTED"]
    out = rag.answer_once(index, None, Q, "self_rag")
    assert any("regenerated" in n for n in out["notes"]), out["notes"]
    chat.judge_replies = ["SUPPORTED"]
    print("Self-RAG regeneration OK")

    # 10. Agentic: the model asks for one more search, then answers
    chat.decisions = ['{"action": "search", "query": "acme headquarters"}', '{"action": "answer"}']
    out = rag.answer_once(index, None, Q, "agentic")
    assert any("Agent searched" in n for n in out["notes"]), out["notes"]
    chat.decisions = ['{"action": "answer"}']
    print("agentic search loop OK")

    # 11. Streaming emits notes before tokens and finishes with done
    events = list(rag.answer_stream(index, graph, Q, "crag"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "sources" and "note" in kinds and kinds[-1] == "done", kinds
    print("streaming events:", kinds)
    print("PIPELINE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
