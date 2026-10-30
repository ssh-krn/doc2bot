"""Offline tests for the knowledge graph: extraction parsing, graph building, local/global/hybrid
retrieval, document removal, persistence, and the RAG retrieval path. A deterministic stub replaces the
LLM, so these tests check graph logic, not extraction quality.
Run from backend/: python -m eval.kg_smoke_test"""

import json
import tempfile
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

from app.ingest import process_document
from app.kg.build import index_document
from app.kg.extract import parse_extraction
from app.kg.graph import KnowledgeGraph
from app.retrieval import Index
from eval.smoke_test import DIM, StubEmbedder

KNOWN = ["Jane Doe", "Acme Corp", "Berlin", "Globex", "Data Summit"]
RELATIONS = [
    ("Jane Doe", "Acme Corp", "is chief scientist at", ["employment", "role"]),
    ("Acme Corp", "Berlin", "is headquartered in", ["location"]),
    ("Globex", "Data Summit", "organizes", ["event"]),
]


def stub_llm(prompt: str) -> str:
    """Deterministic stand-in for the extraction model. Returns JSON, sometimes fenced, like a real model."""
    passage = prompt.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
    present = [n for n in KNOWN if n in passage]
    entities = [{"name": n, "type": "organization" if "Corp" in n or n == "Globex" else "other",
                 "description": f"{n} mentioned in the text"} for n in present]
    rels = [{"source": s, "target": t, "description": d, "keywords": k}
            for s, t, d, k in RELATIONS if s in present and t in present]
    payload = json.dumps({"entities": entities, "relations": rels})
    return f"```json\n{payload}\n```" if "Acme" in passage else payload


def pdf_bytes(pages: list[str]) -> bytes:
    import pymupdf
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def main() -> None:
    emb = StubEmbedder()
    # 1. Parser: garbage, fenced JSON, and relation endpoints that are missing from entities
    assert parse_extraction("no json here") == ([], [])
    assert parse_extraction('{"entities": "oops"}') == ([], [])
    ents, rels = parse_extraction('Sure! ```json {"entities":[],"relations":[{"source":"A","target":"B","description":"x","keywords":[]}]}```')
    assert {e["name"] for e in ents} == {"A", "B"} and len(rels) == 1
    print("parser OK")

    # 2. Ingest two documents, then build the graph from their chunks
    doc_a = pdf_bytes([
        "Jane Doe is chief scientist at Acme Corp.",
        "Acme Corp is headquartered in Berlin.",
    ])
    doc_b = pdf_bytes(["Globex organizes the Data Summit in Berlin."])
    index = Index(emb, QdrantClient(":memory:"), "kg_test", DIM, reranker=None)
    graph = KnowledgeGraph(emb, path=None)

    ids_by_doc = {}
    for name, data in [("people.pdf", doc_a), ("events.pdf", doc_b)]:
        chunks, _ = process_document(name, data, 800, 120)
        doc_id = str(uuid.uuid4())
        ids = index.add_document(doc_id, name, chunks)
        assert len(ids) == len(chunks)
        ids_by_doc[name] = (doc_id, ids, chunks)
        stats = index_document(graph, stub_llm, doc_id, [{"id": i, "text": c["text"]} for i, c in zip(ids, chunks)], workers=2)
        print(name, stats)

    s = graph.stats()
    assert s["entities"] >= 5 and s["relations"] >= 3, s
    print("graph", s)

    jane_doc_id, _, jane_chunks = ids_by_doc["people.pdf"]
    jane_chunk_id = next(i for i, c in zip(ids_by_doc["people.pdf"][1], jane_chunks) if "chief scientist" in c["text"])

    # 3. Edge provenance: the relation must point back to the chunk it came from
    edge = graph.g.edges["jane doe", "acme corp"]
    assert jane_chunk_id in edge["sources"], edge

    # 4. Local mode: entity-centred facts and their chunks
    facts, chunk_ids = graph.context("Who is Jane Doe?", mode="local")
    assert any("Jane Doe" in f for f in facts), facts
    assert jane_chunk_id in chunk_ids, (chunk_ids, jane_chunk_id)
    print("local facts:", facts[:3])

    # 5. Global mode: relation-centred facts. Query and relation text are close in the stub's embedding space
    facts, chunk_ids = graph.context("Acme Corp is headquartered in Berlin", mode="global")
    assert any("headquartered" in f for f in facts), facts
    assert chunk_ids, "global mode should return source chunks"
    print("global facts:", facts[:2])

    # 6. Naive mode uses no graph
    assert graph.context("Who is Jane Doe?", mode="naive") == ([], [])

    # 7. Removing a document drops what only it supported, and keeps shared nodes
    graph.remove_document(jane_doc_id, ids_by_doc["people.pdf"][1])
    assert not graph.g.has_node("jane doe"), "entity unique to the removed document should be gone"
    assert graph.g.has_node("berlin"), "Berlin is still supported by the other document"
    assert graph.g.has_node("globex")
    print("after removal", graph.stats())

    # 8. Persistence round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kg.json"
        g1 = KnowledgeGraph(emb, path=path)
        g1.add_extraction("d", "c1", [{"name": "Acme Corp", "type": "organization", "description": "x"}], [])
        g1.save()
        g2 = KnowledgeGraph(emb, path=path)
        assert g2.stats() == g1.stats() and g2.g.has_node("acme corp")
    print("persistence OK")

    # 9. RAG retrieval: hybrid adds graph facts; naive does not
    from app.rag import retrieve
    index2 = index  # same index, after removal
    _, facts_hybrid = retrieve(index2, graph, "What does Globex organize?", "hybrid")
    _, facts_naive = retrieve(index2, graph, "What does Globex organize?", "naive")
    assert facts_hybrid and not facts_naive
    print("rag retrieval OK; hybrid facts:", facts_hybrid[:2])
    print("KG SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
