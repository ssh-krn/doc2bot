import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .ingest import extract_chunks
from .rag import answer_stream
from .store import get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store()
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


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents():
    return get_store().list_documents()


@app.post("/api/documents", status_code=201)
def upload_document(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    data = file.file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB")

    try:
        chunks, page_count = extract_chunks(data, settings.chunk_size, settings.chunk_overlap)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this PDF")
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. This may be a scanned PDF, which needs OCR.",
        )

    doc_id = str(uuid.uuid4())
    get_store().add_document(doc_id, file.filename, chunks)
    return {"doc_id": doc_id, "filename": file.filename, "pages": page_count, "chunks": len(chunks)}


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    get_store().delete_document(doc_id)


@app.post("/api/chat")
def chat(req: ChatRequest):
    store = get_store()

    def events():
        try:
            for event in answer_stream(store, req.question):
                yield json.dumps(event) + "\n"
        except RuntimeError as e:  # e.g. missing API key
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
