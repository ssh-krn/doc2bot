"""Common output of every parser. Mirrors RAG-Anything's content list: each document
becomes typed blocks (text, table, image, equation) that are processed per modality."""

from dataclasses import dataclass, field
from typing import Literal

Modality = Literal["text", "table", "image", "equation"]


@dataclass
class ContentBlock:
    modality: Modality
    text: str                      # extracted text, table markdown, or image description
    page: int                      # 1-based page, slide, or sheet number
    image_bytes: bytes | None = None
    image_mime: str | None = None
    meta: dict = field(default_factory=dict)


class ParseError(Exception):
    """A file that can't be processed. The message is shown to the user."""
