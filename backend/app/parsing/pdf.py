import pymupdf

from .equations import looks_like_equation
from .tables import table_to_markdown
from .types import ContentBlock

MIN_IMAGE_PX = 120  # skip logos, bullets, and other decoration
_IMAGE_EXT_TO_MIME = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg",
                      "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp"}


def _render(page: pymupdf.Page, rect: pymupdf.Rect) -> bytes:
    clip = pymupdf.Rect(rect.x0 - 2, rect.y0 - 2, rect.x1 + 2, rect.y1 + 2) & page.rect
    return page.get_pixmap(clip=clip, dpi=150).tobytes("png")


def parse_pdf(data: bytes) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    seen_images: set[int] = set()  # the same image xref can appear on several pages

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page_no, page in enumerate(doc, start=1):
            # 1. Tables first, so their text isn't also emitted as plain text.
            table_rects: list[pymupdf.Rect] = []
            try:
                found = page.find_tables().tables
            except Exception:
                found = []
            for table in found:
                md = table_to_markdown(table.extract())
                table_rects.append(pymupdf.Rect(table.bbox))
                if md:
                    blocks.append(ContentBlock("table", md, page_no))

            # 2. Text in reading order (top to bottom, left to right), skipping table regions.
            # Equation-like lines become equation blocks, with a rendered crop for the vision model.
            text_items = []
            for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
                if block_type != 0 or not text.strip():
                    continue  # type 1 is an image block
                rect = pymupdf.Rect(x0, y0, x1, y1)
                if any(rect.intersects(tr) for tr in table_rects):
                    continue
                text_items.append((round(y0), round(x0), text.strip(), rect))
            for _, _, text, rect in sorted(text_items, key=lambda t: (t[0], t[1])):
                if looks_like_equation(text):
                    blocks.append(ContentBlock("equation", text, page_no,
                                               image_bytes=_render(page, rect), image_mime="image/png"))
                else:
                    blocks.append(ContentBlock("text", text, page_no))

            # 3. Embedded images, kept as blocks so they can be described if a vision model is enabled.
            for info_ in page.get_images(full=True):
                xref = info_[0]
                if xref in seen_images:
                    continue
                seen_images.add(xref)
                info = doc.extract_image(xref)
                if not info or min(info["width"], info["height"]) < MIN_IMAGE_PX:
                    continue
                mime = _IMAGE_EXT_TO_MIME.get(info["ext"].lower())
                if mime is None:
                    continue  # e.g. JPX or JBIG2; not supported by common vision models
                blocks.append(ContentBlock("image", "", page_no,
                                           image_bytes=info["image"], image_mime=mime))
    return blocks
