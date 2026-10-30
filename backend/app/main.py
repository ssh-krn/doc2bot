import importlib
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import images
from .config import get_settings
from .ingest import process_document
from .kg.build import index_document, llm_invoke_factory, llm_summarize_factory
from .parsing import SUPPORTED_EXTENSIONS, ParseError, extension_of
from .pipeline.catalog import catalog
from .pipeline.options import MODES, make_options
from .rag import answer_stream
from .store import get_graph, get_store

log = logging.getLogger("rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store()  # load embedding and reranking models once at startup
    get_graph()  # load the persisted graph, if enabled
    yield


settings = get_settings()
app = FastAPI(title="PDF RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str = Field(default="traditional", max_length=32)
    lightrag_mode: str = Field(default="hybrid", max_length=16)
    techniques: list[str] = Field(default_factory=lambda: ["hybrid"], max_length=8)
    rerankers: list[str] = Field(default_factory=lambda: ["cross_encoder"], max_length=4)
    vision: bool | None = None  # None: use ENABLE_VLM. True/False: attach figures to this question or not


def _reranker_availability() -> dict[str, bool]:
    s = get_settings()
    flashrank = importlib.util.find_spec("flashrank") is not None
    cohere = bool(s.cohere_api_key) and importlib.util.find_spec("cohere") is not None
    return {"cross_encoder": True, "bi_encoder": True, "flashrank": flashrank, "cohere": cohere}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/options")
def options():
    """Everything the UI needs to offer: modes, techniques, rerankers and their availability."""
    return catalog(_reranker_availability())


@app.get("/api/documents")
def list_documents():
    return get_store().list_documents()


@app.get("/api/graph/stats")
def graph_stats():
    graph = get_graph()
    if graph is None:
        return {"enabled": False, "entities": 0, "relations": 0, "communities": 0}
    return {"enabled": True, "communities": len(graph.communities), **graph.stats()}


@app.post("/api/documents", status_code=201)
def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    if extension_of(filename) not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported: PDF, DOC, DOCX, PPT, PPTX, XLS, XLSX, and common image formats.",
        )
    data = file.file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB")

    try:
        chunks, page_count = process_document(filename, data, settings.chunk_size, settings.chunk_overlap)
    except ParseError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        log.exception("failed to process %s", filename)
        raise HTTPException(status_code=400, detail="Could not read this file")

    doc_id = str(uuid.uuid4())
    store = get_store()
    chunk_ids = store.add_document(doc_id, filename, chunks)

    kg_stats = None
    graph = get_graph()
    if graph is not None:
        try:
            kg_stats = index_document(
                graph, llm_invoke_factory(), doc_id,
                [{"id": cid, "text": c["text"]} for cid, c in zip(chunk_ids, chunks)],
                workers=settings.kg_workers,
            )
            if settings.enable_communities:
                graph.refresh_communities(summarize=llm_summarize_factory())
                kg_stats["communities"] = len(graph.communities)
        except Exception:
            # The document is still searchable by chunks; only graph retrieval is missing for it.
            log.exception("knowledge graph indexing failed for %s", filename)
            kg_stats = {"error": "graph indexing failed; chunk search still works"}

    modalities = {m: sum(1 for c in chunks if c["modality"] == m) for m in {c["modality"] for c in chunks}}
    return {"doc_id": doc_id, "filename": filename, "kind": extension_of(filename)[1:],
            "pages": page_count, "chunks": len(chunks), "modalities": modalities, "graph": kg_stats}


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    store = get_store()
    chunk_ids = store.chunk_ids_for_doc(doc_id)
    for key in store.image_keys_for_doc(doc_id):
        images.delete(key)
    store.delete_document(doc_id)
    graph = get_graph()
    if graph is not None and chunk_ids:
        graph.remove_document(doc_id, chunk_ids)
        graph.save()
        if settings.enable_communities:
            try:
                # Summaries are cached by cluster, so only clusters that changed are summarized again.
                graph.refresh_communities(summarize=llm_summarize_factory())
            except Exception:
                log.exception("community refresh failed after delete")
                graph.refresh_communities(summarize=None)


@app.post("/api/chat")
def chat(req: ChatRequest):
    store = get_store()
    graph = get_graph()
    try:
        make_options(mode=req.mode, lightrag_mode=req.lightrag_mode, techniques=req.techniques,
                     rerankers=req.rerankers, vision=req.vision)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    available = _reranker_availability()
    missing = [r for r in req.rerankers if not available.get(r, False)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Reranker not available: {', '.join(missing)}")

    def events():
        try:
            for event in answer_stream(store, graph, req.question, req.mode, req.vision,
                                       techniques=req.techniques, rerankers=req.rerankers,
                                       lightrag_mode=req.lightrag_mode):
                yield json.dumps(event) + "\n"
        except Exception as e:  # e.g. missing API key, reranker error
            log.exception("chat failed")
            yield json.dumps({"type": "error", "data": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
