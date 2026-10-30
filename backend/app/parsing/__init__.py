"""Format dispatch. parse_file turns any supported upload into ContentBlocks."""

from pathlib import Path

from .image import parse_image
from .office import parse_office
from .pdf import parse_pdf
from .types import ContentBlock, ParseError

PDF_EXTENSIONS = {".pdf"}
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".xlsm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | OFFICE_EXTENSIONS | IMAGE_EXTENSIONS


def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def parse_file(filename: str, data: bytes) -> list[ContentBlock]:
    ext = extension_of(filename)
    if ext in PDF_EXTENSIONS:
        return parse_pdf(data)
    if ext in OFFICE_EXTENSIONS:
        return parse_office(ext, data)
    if ext in IMAGE_EXTENSIONS:
        return parse_image(ext, data)
    raise ParseError(f"Unsupported file type: {ext or 'no extension'}")


__all__ = ["ContentBlock", "ParseError", "parse_file", "extension_of", "SUPPORTED_EXTENSIONS",
           "IMAGE_EXTENSIONS"]
