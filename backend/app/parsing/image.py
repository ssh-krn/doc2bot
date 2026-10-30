import io

from PIL import Image, UnidentifiedImageError

from .types import ContentBlock, ParseError

_WEB_SAFE = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".webp": "image/webp", ".gif": "image/gif"}


def parse_image(ext: str, data: bytes) -> list[ContentBlock]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ParseError(f"Could not read this image: {e}")

    if ext in _WEB_SAFE:
        return [ContentBlock("image", "", 1, image_bytes=data, image_mime=_WEB_SAFE[ext])]
    # BMP, TIFF and others: re-encode to PNG so vision models can read them.
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return [ContentBlock("image", "", 1, image_bytes=buf.getvalue(), image_mime="image/png")]
