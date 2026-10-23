"""Image preparation shared by every LLM-calling step."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image

# Long-edge cap for the image sent to the model. Vision pricing is flat up
# to roughly 1500 px on most providers; the default keeps a little headroom
# so small dimension text stays legible.
MAX_IMAGE_SIDE = 3000
JPEG_QUALITY = 85


def prepare_image(path: Path, max_side: int = MAX_IMAGE_SIDE,
                  quality: int = JPEG_QUALITY) -> tuple[bytes, str]:
    """Open ``path``, downscale so the long edge is at most ``max_side`` and
    re-encode as JPEG. Returns ``(jpeg_bytes, media_type)``.

    Colour is preserved: some sheets have photographic or branded
    backgrounds and vision models handle those better than a thresholded
    silhouette would.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), "image/jpeg"


def prepare_pil(img: Image.Image, max_side: int = MAX_IMAGE_SIDE,
                quality: int = JPEG_QUALITY) -> tuple[bytes, str]:
    """Same as :func:`prepare_image` for an already-open PIL image."""
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), "image/jpeg"


def sent_dims(img_bytes: bytes) -> tuple[int, int]:
    """``(height, width)`` of the image bytes as actually encoded and sent."""
    with Image.open(io.BytesIO(img_bytes)) as im:
        w, h = im.size
    return h, w


def sha256_file(path: Path | str) -> str:
    """Streamed SHA-256 of a file. Used as the content-addressed image id."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
