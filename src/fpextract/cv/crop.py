"""LLM coarse localisation → OpenCV boundary snapping.

The vision model returns a rough ``crop_frac`` per unit. Inside that box
(plus a safety pad) OpenCV finds the tightest bounding box of actual
drawing content and adds a small margin. Works on white, dark and branded
backgrounds by sampling the corners of the rough crop to pick a threshold
strategy before contouring.
"""
from __future__ import annotations

from PIL import Image

from fpextract.cv import require_cv2


def pad_frac(frac: list[float], pad: float) -> list[float]:
    """Expand a ``[x0, y0, x1, y1]`` fraction box by ``pad`` on every edge."""
    return [max(0.0, frac[0] - pad), max(0.0, frac[1] - pad),
            min(1.0, frac[2] + pad), min(1.0, frac[3] + pad)]


def crop_frac_pil(img: Image.Image, frac: list[float], pad: float = 0.0) -> Image.Image:
    w, h = img.size
    f = pad_frac(frac, pad)
    x1, y1, x2, y2 = int(f[0] * w), int(f[1] * h), int(f[2] * w), int(f[3] * h)
    return img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))


def snap_to_content(crop_bgr, margin_frac: float = 0.015) -> tuple[int, int, int, int]:
    """Tightest content box ``(x0, y0, x1, y1)`` in pixel coords of ``crop_bgr``."""
    require_cv2()
    import cv2
    import numpy as np

    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    s = max(2, min(h, w) // 30)
    corners = np.concatenate([gray[:s, :s].ravel(), gray[:s, -s:].ravel(),
                              gray[-s:, :s].ravel(), gray[-s:, -s:].ravel()])
    bg = float(np.median(corners))
    if bg > 200:
        _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    elif bg < 80:
        edges = cv2.Canny(gray, 20, 60)
        _, bright = cv2.threshold(gray, min(bg + 40, 200), 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(edges, bright)
    else:
        mask = cv2.Canny(gray, 30, 100)
    k = max(5, min(h, w) // 15) | 1
    dilated = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h
    # The dilated contour only picks the main connected blob; the box itself is
    # measured on the undilated mask inside it so the result is tight.
    rx, ry, rw, rh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    sub = mask[ry:ry + rh, rx:rx + rw]
    ys, xs = np.where(sub > 0)
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1 = rx + int(xs.min()), rx + int(xs.max()) + 1
    y0, y1 = ry + int(ys.min()), ry + int(ys.max()) + 1
    mx, my = max(1, int(w * margin_frac)), max(1, int(h * margin_frac))
    return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)


def refine_crop(img: Image.Image, frac: list[float], pad: float = 0.02,
                margin_frac: float = 0.015) -> list[float]:
    """Return the CV-refined ``crop_frac`` for an LLM box on the full image."""
    require_cv2()
    import cv2
    import numpy as np

    W, H = img.size
    padded = pad_frac(frac, pad)
    rough = crop_frac_pil(img, frac, pad)
    if rough.size[0] < 2 or rough.size[1] < 2:
        return list(frac)
    bgr = cv2.cvtColor(np.array(rough.convert("RGB")), cv2.COLOR_RGB2BGR)
    rx0, ry0 = int(padded[0] * W), int(padded[1] * H)
    cx1, cy1, cx2, cy2 = snap_to_content(bgr, margin_frac=margin_frac)
    return [(rx0 + cx1) / W, (ry0 + cy1) / H, (rx0 + cx2) / W, (ry0 + cy2) / H]
