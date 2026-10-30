"""Stores the bytes of extracted figures, so a question can send the actual image to a vision model.
Files live in IMAGE_DIR, named by a random key that the chunk index keeps in its payload."""

import uuid
from pathlib import Path

from .config import get_settings

_EXT_TO_MIME = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}
_MIME_TO_EXT = {v: k for k, v in _EXT_TO_MIME.items()}


def _dir() -> Path:
    d = Path(get_settings().image_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(data: bytes, mime: str) -> str:
    key = uuid.uuid4().hex
    (_dir() / f"{key}.{_MIME_TO_EXT.get(mime, 'png')}").write_bytes(data)
    return key


def load(key: str) -> tuple[bytes, str] | None:
    for path in _dir().glob(f"{key}.*"):
        mime = _EXT_TO_MIME.get(path.suffix.lstrip("."), "image/png")
        return path.read_bytes(), mime
    return None


def delete(key: str) -> None:
    for path in _dir().glob(f"{key}.*"):
        path.unlink(missing_ok=True)
