"""Tiled text-watermark removal without model weights.

Many marketing sheets carry a semi-transparent brand word repeated in a
regular grid. This module locates it with template matching and inpaints
only the letter shapes:

1. Background normalisation — median-blur subtract, keep only the watermark
   brightness band — so watermarks on cream or tinted backgrounds look alike.
2. Template extraction — the leftmost word-sized run of text in the top
   quarter of the normalised image becomes the per-image template.
3. Template matching + non-max suppression at ``match_threshold``.
4. Grid projection — infer the watermark grid and fill in instances cropped
   at the image borders.
5. Letter-shape stencil stamped at every site; OpenCV TELEA inpaint.

Returns the input unchanged when no plausible watermark is found, so it is
safe to run on a mixed batch. Pure CPU.
"""
from __future__ import annotations

from typing import Any

from fpextract.cv import require_cv2

DEFAULT_BG_BLUR_KSIZE = 101
DEFAULT_WM_DARKNESS_MIN = 6
DEFAULT_WM_DARKNESS_MAX = 35
DEFAULT_MATCH_THRESHOLD = 0.35
DEFAULT_STENCIL_DILATE = 1
# Below this many hits the auto-template was probably project text, not a
# repeated watermark (which comes in grids of 15–30 instances).
DEFAULT_MIN_MATCHES = 5
# Plausible watermark word width in pixels on a typical ~2000 px sheet.
DEFAULT_TEMPLATE_WIDTH = (150, 320)


def _skip(reason: str, n_detected: int = 0) -> dict[str, Any]:
    return {"matched": False, "n_matches": n_detected, "n_detected": n_detected,
            "n_projected": 0, "skipped": reason}


def _runs(flags) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def remove_tiled_watermark(img_rgb, *, bg_blur_ksize: int = DEFAULT_BG_BLUR_KSIZE,
                           wm_darkness_min: int = DEFAULT_WM_DARKNESS_MIN,
                           wm_darkness_max: int = DEFAULT_WM_DARKNESS_MAX,
                           match_threshold: float = DEFAULT_MATCH_THRESHOLD,
                           stencil_dilate: int = DEFAULT_STENCIL_DILATE,
                           min_matches: int = DEFAULT_MIN_MATCHES,
                           template_width: tuple[int, int] = DEFAULT_TEMPLATE_WIDTH,
                           ) -> tuple[Any, dict[str, Any]]:
    """``(clean_rgb, info)``; ``info["matched"]`` says whether anything changed."""
    require_cv2()
    import cv2
    import numpy as np

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image (H, W, 3); got shape {img_rgb.shape}")
    H, W = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. background normalisation
    if bg_blur_ksize % 2 == 0:
        bg_blur_ksize += 1
    local_bg = cv2.medianBlur(gray, bg_blur_ksize)
    diff = local_bg.astype(np.int32) - gray.astype(np.int32)
    in_band = (diff >= wm_darkness_min) & (diff <= wm_darkness_max)
    normalized = np.where(in_band, diff, 0).astype(np.float32)
    normalized = (normalized / wm_darkness_max * 255).clip(0, 255).astype(np.uint8)

    # 2. template extraction (leftmost word-sized run in the top quarter)
    top_band = (normalized[: H // 4, :] > 60).astype(np.uint8)
    runs = _runs(top_band.sum(axis=0) > 3)
    words: list[tuple[int, int]] = []
    if runs:
        cs, ce = runs[0]
        for s, e in runs[1:]:
            if s - ce < 25:
                ce = e
            else:
                words.append((cs, ce))
                cs, ce = s, e
        words.append((cs, ce))
    lo_w, hi_w = template_width
    candidates = [w for w in words if lo_w < w[1] - w[0] < hi_w]
    if not candidates:
        return img_rgb.copy(), _skip("no_template_candidate")
    tpl_x0, tpl_x1 = candidates[0]

    strip = top_band[:, tpl_x0:tpl_x1]
    blocks = [(a, b) for a, b in _runs(strip.sum(axis=1) > 20) if b - a >= 10]
    if not blocks:
        return img_rgb.copy(), _skip("no_template_vertical")
    y_start, y_end = max(blocks, key=lambda ab: ab[1] - ab[0])  # tallest block wins
    tpl_y0, tpl_y1 = max(0, y_start - 3), min(H // 4, y_end + 3)
    template = normalized[tpl_y0:tpl_y1, tpl_x0:tpl_x1]
    th, tw = template.shape
    if th < 10 or tw < 30:
        return img_rgb.copy(), _skip("template_too_small")

    # 3. template matching + NMS
    result = cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)
    locs = np.where(result >= match_threshold)
    cands = sorted(((int(x), int(y), float(result[y, x])) for y, x in zip(*locs, strict=True)), key=lambda p: -p[2])
    kept: list[tuple[int, int, float]] = []
    for p in cands:
        if not any(abs(p[0] - k[0]) < tw * 0.4 and abs(p[1] - k[1]) < th * 0.6 for k in kept):
            kept.append(p)
    if len(kept) < min_matches:
        return img_rgb.copy(), _skip(f"too_few_matches({len(kept)}<{min_matches})", len(kept))

    # 4. grid projection
    boxes = [(x, y) for x, y, _ in kept]
    clusters: list[list[int]] = []
    for ry in sorted({y for _, y in boxes}):
        for c in clusters:
            if abs(c[0] - ry) < th * 0.6:
                c.append(ry)
                break
        else:
            clusters.append([ry])
    rows = []
    for c in clusters:
        ry = int(np.mean(c))
        rows.append((ry, sorted(x for x, y in boxes if abs(y - ry) < th * 0.6)))
    diffs = [b - a for _, xs in rows if len(xs) >= 2 for a, b in zip(xs, xs[1:], strict=False)]
    spacing = int(np.median(diffs)) if diffs else tw
    projected: list[tuple[int, int]] = []
    for ry, xs in rows:
        if not xs:
            continue
        cur = xs[0]
        while cur < W:
            if not any(abs(cur - x) < tw * 0.3 for x in xs):
                projected.append((cur, ry))
            cur += spacing
        cur = xs[0] - spacing
        while cur + tw > 0:
            if not any(abs(cur - x) < tw * 0.3 for x in xs):
                projected.append((cur, ry))
            cur -= spacing

    # 5. stencil + 6. inpaint
    stencil = (template > 80).astype(np.uint8) * 255
    if stencil_dilate > 0:
        stencil = cv2.dilate(stencil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                             iterations=stencil_dilate)
    mask = np.zeros((H, W), dtype=np.uint8)
    for x, y in boxes + projected:
        sx0, sy0 = max(0, -x), max(0, -y)
        sx1, sy1 = min(tw, W - x), min(th, H - y)
        dx0, dy0 = max(0, x), max(0, y)
        dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
        if sx1 > sx0 and sy1 > sy0:
            mask[dy0:dy1, dx0:dx1] = np.maximum(mask[dy0:dy1, dx0:dx1], stencil[sy0:sy1, sx0:sx1])
    clean_bgr = cv2.inpaint(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB), {
        "matched": True, "n_matches": len(boxes) + len(projected),
        "n_detected": len(boxes), "n_projected": len(projected), "skipped": None,
    }
