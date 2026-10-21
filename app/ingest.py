import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_chunks(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> tuple[list[dict], int]:
    """Return (chunks, page_count). Each chunk has text, page (1-based), chunk_index."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = doc.page_count
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text:
                continue  # scanned or image-only page; no OCR in this version
            for piece in splitter.split_text(text):
                chunks.append({"text": piece, "page": page_no, "chunk_index": len(chunks)})
    return chunks, page_count
