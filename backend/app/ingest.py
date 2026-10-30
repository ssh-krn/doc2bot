"""Document ingestion, following RAG-Anything's multi-stage pipeline:

1. Parse the file into typed content blocks (text, table, image, equation).
2. Run two pipelines concurrently:
   - text pipeline: text and tables, chunked per page with overlap; tables keep their header when split
   - multimodal pipeline: vision-model work on figures and equations (descriptions, LaTeX), then chunking
3. Merge the chunks in document order.

Figure bytes are kept in the image store, so a question can later send the figure itself to a vision model.
"""

from concurrent.futures import ThreadPoolExecutor

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import images
from .config import get_settings
from .parsing import IMAGE_EXTENSIONS, ParseError, extension_of, parse_file
from .parsing.pdf import parse_pdf
from .parsing.types import ContentBlock


def _split_table(text: str, size: int) -> list[str]:
    """Split a Markdown table into pieces under `size`, repeating the header in each piece."""
    lines = text.split("\n")
    sep = next((i for i, line in enumerate(lines) if line.startswith("|---")), None)
    if sep is None:
        return [text[i:i + size] for i in range(0, len(text), size)]
    header, body = "\n".join(lines[: sep + 1]), lines[sep + 1:]
    pieces, current = [], []
    for row in body:
        candidate = "\n".join([header, *current, row])
        if current and len(candidate) > size:
            pieces.append("\n".join([header, *current]))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join([header, *current]))
    return pieces


def blocks_to_chunks(blocks: list[ContentBlock], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[dict] = []

    def add(text: str, page: int, modality: str, image_key: str | None = None) -> None:
        chunk = {"text": text, "page": page, "chunk_index": len(chunks), "modality": modality}
        if image_key:
            chunk["image_key"] = image_key
        chunks.append(chunk)

    pending: list[str] = []
    pending_page: int | None = None

    def flush_text() -> None:
        nonlocal pending, pending_page
        if pending and pending_page is not None:
            for piece in splitter.split_text("\n\n".join(pending)):
                add(piece, pending_page, "text")
        pending, pending_page = [], None

    for block in blocks:
        if block.modality == "text":
            if pending_page is not None and block.page != pending_page:
                flush_text()
            pending_page = block.page
            pending.append(block.text)
            continue

        flush_text()
        if not block.text.strip():
            continue  # e.g. a figure with no description: nothing to search
        image_key = None
        if block.modality == "image" and block.image_bytes is not None:
            image_key = images.save(block.image_bytes, block.image_mime or "image/png")

        if block.modality == "table" and len(block.text) > chunk_size:
            for piece in _split_table(block.text, chunk_size):
                add(piece, block.page, "table")
        elif len(block.text) > chunk_size:
            for piece in splitter.split_text(block.text):
                add(piece, block.page, block.modality, image_key)
        else:
            add(block.text, block.page, block.modality, image_key)

    flush_text()
    return chunks


def assign_parents(chunks: list[dict], group_size: int) -> list[dict]:
    """Groups consecutive text and table chunks from one page into parents. Figures and equations are
    their own parents. Parent-document and auto-merging retrievers use these groups."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for c in chunks:
        if c["modality"] in ("text", "table"):
            starts_new = (
                current and (current[-1]["page"] != c["page"] or len(current) >= group_size)
            )
            if starts_new:
                groups.append(current)
                current = []
            current.append(c)
        else:
            if current:
                groups.append(current)
                current = []
            groups.append([c])
    if current:
        groups.append(current)
    for gi, group in enumerate(groups):
        parent_text = "\n\n".join(c["text"] for c in group)[:4000]
        for c in group:
            c["parent_id"] = f"p{gi}"
            c["parent_text"] = parent_text
    return chunks


def _multimodal_pipeline(blocks: list[ContentBlock], chunk_size: int, chunk_overlap: int) -> list[dict]:
    if get_settings().enable_vlm and blocks:
        from .parsing.vision import enrich_blocks  # imported lazily: needs a vision provider key
        enrich_blocks(blocks)
    return blocks_to_chunks(blocks, chunk_size, chunk_overlap)


def process_document(filename: str, data: bytes, chunk_size: int, chunk_overlap: int) -> tuple[list[dict], int]:
    """Returns (chunks, page_count). Raises ParseError with a user-facing message."""
    s = get_settings()
    if extension_of(filename) in IMAGE_EXTENSIONS and not s.enable_vlm:
        raise ParseError("Image uploads need ENABLE_VLM=true, since images are made searchable by a vision model.")

    blocks = parse_file(filename, data)
    text_blocks = [b for b in blocks if b.modality in ("text", "table")]
    mm_blocks = [b for b in blocks if b.modality in ("image", "equation")]

    with ThreadPoolExecutor(max_workers=2) as pool:
        text_future = pool.submit(blocks_to_chunks, text_blocks, chunk_size, chunk_overlap)
        mm_future = pool.submit(_multimodal_pipeline, mm_blocks, chunk_size, chunk_overlap)
        chunks = text_future.result() + mm_future.result()

    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i
    assign_parents(chunks, get_settings().parent_group_size)
    pages = max((b.page for b in blocks), default=0)
    if not chunks:
        raise ParseError("No searchable text found. The file may be scanned, or contain only images.")
    return chunks, pages


def extract_chunks(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> tuple[list[dict], int]:
    """PDF-only entry point kept for the evaluation harness (no vision calls)."""
    blocks = parse_pdf(pdf_bytes)
    chunks = blocks_to_chunks(blocks, chunk_size, chunk_overlap)
    pages = max((b.page for b in blocks), default=0)
    return chunks, pages
