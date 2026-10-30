"""Offline tests for multimodal ingestion and VLM-enhanced queries. Vision and text models are stubbed,
so this checks the pipeline logic, not how well a real model reads figures or formulas.
Run from backend/: python -m eval.multimodal_smoke_test"""

import os
import tempfile
import uuid
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["IMAGE_DIR"] = str(Path(_tmp) / "images")
os.environ["ENABLE_VLM"] = "true"
os.environ["MISTRAL_API_KEY"] = "test-key"
os.environ["VLM_WORKERS"] = "2"

import pymupdf  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app import images  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingest import process_document  # noqa: E402
from app.parsing.equations import looks_like_equation  # noqa: E402
from app.retrieval import Index  # noqa: E402
import app.parsing.vision as vision_module  # noqa: E402
import app.rag as rag_module  # noqa: E402
from eval.smoke_test import DIM, StubEmbedder  # noqa: E402


class FakeVision:
    """Answers description prompts with a figure description and LaTeX prompts with a formula."""
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        content = messages[-1].content
        text = next(p["text"] for p in content if p["type"] == "text")
        self.calls.append(messages)
        if "LaTeX" in text:
            return AIMessage(content="E = mc^{2}")
        return AIMessage(content="A bar chart of revenue by quarter: Q1 10 million, Q2 20 million.")


class FakeText:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="Revenue rose in Q2 [1].", usage_metadata={"input_tokens": 9, "output_tokens": 5, "total_tokens": 14})

    def stream(self, messages):
        self.calls.append(messages)
        yield AIMessage(content="Revenue rose in Q2 [1].")


def make_pdf() -> bytes:
    # A 200x150 red figure, large enough to pass the figure-size filter
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 150), False)
    pix.set_rect(pix.irect, (220, 30, 30))
    png = pix.tobytes("png")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Refund policy: returns within 30 days of purchase.", fontsize=11)
    page.insert_text((72, 110), "E = mc^2", fontsize=12)
    page.insert_image(pymupdf.Rect(72, 150, 272, 300), stream=png)
    data = doc.tobytes()
    doc.close()
    return data


def has_image_part(messages) -> bool:
    content = messages[-1].content
    return isinstance(content, list) and any(p.get("type") == "image_url" for p in content)


def main():
    assert looks_like_equation("E = mc^2")

    fake_vision, fake_text = FakeVision(), FakeText()
    vision_module.vision_llm = lambda: fake_vision  # stubs must be in place before any ingestion
    rag_module.get_llm = lambda: fake_text

    pdf = make_pdf()
    chunks, pages = process_document("report.pdf", pdf, 800, 120)
    modalities = {c["modality"] for c in chunks}
    print("modalities:", sorted(modalities), "chunks:", len(chunks))
    assert {"text", "equation", "image"} <= modalities, modalities
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))

    eq = next(c for c in chunks if c["modality"] == "equation")
    assert "E = mc^{2}" in eq["text"] and "E = mc^2" in eq["text"], eq["text"]  # LaTeX plus raw text

    fig = next(c for c in chunks if c["modality"] == "image")
    assert "bar chart" in fig["text"].lower() and fig.get("image_key")
    loaded = images.load(fig["image_key"])
    assert loaded is not None and loaded[1] == "image/png" and loaded[0][:4] == b"\x89PNG"
    print("figure stored:", fig["image_key"][:8], len(loaded[0]), "bytes")

    # Concurrent pipelines must still keep the text chunk
    assert any("Refund policy" in c["text"] for c in chunks)

    # Index it, then ask a question that only the figure answers
    emb = StubEmbedder()
    index = Index(emb, QdrantClient(":memory:"), "mm", DIM, reranker=None)
    index.add_document(str(uuid.uuid4()), "report.pdf", chunks)

    sources, _ = rag_module.retrieve(index, None, "bar chart revenue quarter", "naive")
    assert any(s.get("image_key") == fig["image_key"] for s in sources), sources
    print("retrieved figure as source:", [s["modality"] for s in sources])

    out = rag_module.answer_once(index, None, "What was revenue by quarter?", "naive", vision=True)
    assert has_image_part(fake_vision.calls[-1]), "figure should be attached to the vision call"
    assert not fake_text.calls, "text model should not be used when figures are attached"
    print("VLM-enhanced answer:", out["answer"])

    # Turning vision off sends text only, even when figures were retrieved
    fake_vision.calls.clear()
    out = rag_module.answer_once(index, None, "What was revenue by quarter?", "naive", vision=False)
    assert not fake_vision.calls and fake_text.calls and not has_image_part(fake_text.calls[-1])
    print("text-only answer OK")

    # Deleting figure files
    images.delete(fig["image_key"])
    assert images.load(fig["image_key"]) is None

    # With VLM off, figures are skipped and equations keep their raw text
    os.environ["ENABLE_VLM"] = "false"
    get_settings.cache_clear()
    chunks_off, _ = process_document("report.pdf", pdf, 800, 120)
    assert not any(c["modality"] == "image" for c in chunks_off)
    assert any(c["modality"] == "equation" and c["text"] == "E = mc^2" for c in chunks_off)
    print("VLM-off ingestion OK")
    os.environ["ENABLE_VLM"] = "true"
    get_settings.cache_clear()

    print("MULTIMODAL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
