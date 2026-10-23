"""Bounding-box coordinate systems emitted by different vision models.

* ``fraction`` — ``[0, 1]`` floats of the image dimensions (Claude and most
  instruction-following models when asked for fractions).
* ``0-1000`` — Qwen2.5-VL style normalised integers; divide by 1000.
* ``qwen3`` — Qwen3-VL absolute pixels of the model's internally
  *smart-resized* image; the denominator comes from :func:`qwen_smart_resize`.

Every path converts to ``crop_frac`` fractions so downstream code never
has to know which model produced a record.
"""
from __future__ import annotations

COORD_SYSTEMS = ("fraction", "0-1000", "qwen3")


def qwen_smart_resize(h: int, w: int, min_pixels: int = 256 * 28 * 28,
                      max_pixels: int = 1280 * 28 * 28, patch: int = 28) -> tuple[int, int]:
    """Replicate Qwen-VL's ``smart_resize`` so the pixel space the model
    reports coordinates in can be reconstructed from the sent image size."""
    pixels = h * w
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        h = round(h * scale / patch) * patch
        w = round(w * scale / patch) * patch
    elif pixels < min_pixels:
        scale = (min_pixels / pixels) ** 0.5
        h = round(h * scale / patch) * patch
        w = round(w * scale / patch) * patch
    return h, w


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def bbox_to_crop_frac(bbox: list, coord: str, sent_h: int, sent_w: int) -> list[float]:
    """Convert a model bbox ``[x0, y0, x1, y1]`` in ``coord`` space to fractions."""
    if coord not in COORD_SYSTEMS:
        raise ValueError(f"unknown coord system {coord!r}; choose one of {COORD_SYSTEMS}")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if coord == "fraction":
        return [_clamp(x0), _clamp(y0), _clamp(x1), _clamp(y1)]
    if coord == "0-1000":
        return [_clamp(x0 / 1000), _clamp(y0 / 1000), _clamp(x1 / 1000), _clamp(y1 / 1000)]
    sh, sw = qwen_smart_resize(sent_h, sent_w)
    return [_clamp(x0 / sw), _clamp(y0 / sh), _clamp(x1 / sw), _clamp(y1 / sh)]


def normalise_record_bboxes(record: dict, coord: str, sent_h: int, sent_w: int) -> dict:
    """Rewrite ``units[].bbox`` (or ``bbox_2d``) into ``units[].crop_frac`` in place."""
    for unit in record.get("units") or []:
        raw = unit.pop("bbox", None) or unit.pop("bbox_2d", None)
        if raw and len(raw) == 4:
            unit["crop_frac"] = bbox_to_crop_frac(raw, coord, sent_h, sent_w)
    return record
