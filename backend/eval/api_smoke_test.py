"""End-to-end API test with offline stand-ins for the embedding model, reranker, and LLM.
Exercises upload (chunking + graph build), graph stats, chat in every mode, and delete.
Run from backend/: python -m eval.api_smoke_test"""

import json
import os
import sys
import tempfile
import types
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["QDRANT_PATH"] = str(Path(_tmp) / "qdrant")
os.environ["KG_PATH"] = str(Path(_tmp) / "kg.json")
os.environ["ENABLE_KG"] = "true"
os.environ["GROQ_API_KEY"] = "test-key"  # the stub LLM never calls the network

from eval.smoke_test import StubEmbedder  # noqa: E402
from eval.kg_smoke_test import pdf_bytes, stub_llm  # noqa: E402

# Replace the model loaders before app modules import them
fake_st = types.ModuleType("sentence_transformers")
class FakeCrossEncoder:
    def __init__(self, name): pass
    def predict(self, pairs):
        return [float(len(set(q.lower().split()) & set(t.lower().split()))) for q, t in pairs]
fake_st.SentenceTransformer = lambda name: StubEmbedder()
fake_st.CrossEncoder = FakeCrossEncoder
sys.modules["sentence_transformers"] = fake_st

from langchain_core.messages import AIMessage  # noqa: E402


class FakeLLM:
    """Answers extraction prompts with graph JSON and question prompts with a cited answer."""

    def invoke(self, messages):
        text = messages[-1].content if isinstance(messages, list) else messages
        if "You extract a knowledge graph" in text:
            return AIMessage(content=stub_llm(text))
        return AIMessage(content="Jane Doe is chief scientist at Acme Corp [1].",
                         usage_metadata={"input_tokens": 120, "output_tokens": 12, "total_tokens": 132})

    def stream(self, messages):
        for piece in ["Jane Doe ", "is chief scientist ", "at Acme Corp [1]."]:
            yield AIMessage(content=piece)


import app.llm as llm_module  # noqa: E402
import app.rag as rag_module  # noqa: E402
import app.kg.build as kg_build  # noqa: E402
llm_module.get_llm = lambda: FakeLLM()
rag_module.get_llm = lambda: FakeLLM()
kg_build.get_llm = lambda: FakeLLM()

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


def main():
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/graph/stats").json()["enabled"] is True

        doc = pdf_bytes(["Jane Doe is chief scientist at Acme Corp.", "Acme Corp is headquartered in Berlin."])
        r = client.post("/api/documents", files={"file": ("people.pdf", doc, "application/pdf")})
        assert r.status_code == 201, r.text
        body = r.json()
        print("upload:", body["chunks"], "chunks", body["graph"])
        assert body["graph"]["relations_extracted"] >= 2
        doc_id = body["doc_id"]

        stats = client.get("/api/graph/stats").json()
        assert stats["entities"] >= 3 and stats["relations"] >= 2, stats

        # Chat in every mode; the stream must contain sources, then tokens, then done
        for mode in ["naive", "local", "global", "hybrid"]:
            r = client.post("/api/chat", json={"question": "Who is Jane Doe?", "mode": mode})
            events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
            types_seen = [e["type"] for e in events]
            assert types_seen[0] == "sources" and types_seen[-1] == "done", (mode, types_seen)
            assert "token" in types_seen, (mode, types_seen)
            has_facts = "facts" in types_seen
            assert has_facts == (mode != "naive"), (mode, types_seen)
            answer = "".join(e["data"] for e in events if e["type"] == "token")
            print(f"chat mode={mode}: events={types_seen} answer={answer!r}")

        # Bad mode is rejected by validation
        assert client.post("/api/chat", json={"question": "x", "mode": "magic"}).status_code == 400

        # Image without VLM is rejected clearly
        from PIL import Image
        import io
        buf = io.BytesIO(); Image.new("RGB", (30, 30), (0, 0, 0)).save(buf, "PNG")
        r = client.post("/api/documents", files={"file": ("photo.png", buf.getvalue(), "image/png")})
        assert r.status_code == 422 and "ENABLE_VLM" in r.json()["detail"], r.text

        # Delete removes the document's graph contribution
        assert client.delete(f"/api/documents/{doc_id}").status_code == 204
        stats = client.get("/api/graph/stats").json()
        assert stats["entities"] == 0 and stats["relations"] == 0, stats
        assert client.get("/api/documents").json() == []
        print("delete cleaned graph:", stats)
    print("API SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
