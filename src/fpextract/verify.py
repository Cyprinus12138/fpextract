"""One-shot visual verifiers. Each renders a self-contained HTML page.

* :func:`try_one` — LLM extraction on one image: crop boxes overlaid, each
  unit's crop beside its extracted rooms, raw JSON.
* :func:`try_crop` — two-pass crop: LLM box (orange) vs CV-refined box (green).
* :func:`try_norm` — background normalisation and optional watermark removal.

Nothing is written to the pipeline JSONL files.
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
import webbrowser
from pathlib import Path
from xml.sax.saxutils import escape as esc

from PIL import Image, ImageDraw, ImageFont

from fpextract.images import prepare_pil, sha256_file
from fpextract.llm.client import LLMClient, load_extraction_schema, load_prompt

_CROP_COLORS = ["#e05555", "#4a9fe0", "#4ac77a", "#e0a03a", "#a04ae0", "#e04aaa", "#4ae0d4", "#c0c040"]

_CSS = """
  body  { font-family: sans-serif; margin: 0; background: #1a1a1a; color: #eee; }
  h1    { font-size: 1rem; padding: 10px 16px; margin: 0; background: #111; border-bottom: 1px solid #333; }
  .meta { font-size: .78rem; color: #aaa; padding: 6px 16px 10px; background: #111; border-bottom: 1px solid #333; }
  .grid { display: flex; flex-wrap: wrap; gap: 14px; padding: 14px; }
  .panel { background: #262626; border-radius: 6px; overflow: hidden; flex: 1 1 420px; min-width: 280px; }
  .wide { flex-basis: 100%; }
  .panel h2 { font-size: .82rem; margin: 0; padding: 7px 12px; background: #333; color: #ccc; }
  .inner { padding: 10px; }
  img { max-width: 100%; height: auto; display: block; border-radius: 4px; }
  pre { font-size: .72rem; white-space: pre-wrap; color: #9cf; background: #1e1e1e; padding: 10px;
        border-radius: 4px; overflow: auto; max-height: 420px; }
  .note { font-size: .72rem; color: #aaa; margin-top: 6px; }
  .sep { width: 100%; flex-basis: 100%; border-top: 1px solid #444; font-size: .8rem; color: #888; padding: 6px 0 2px; }
  .llm { color: #e08030; font-weight: bold; } .cv { color: #40c060; font-weight: bold; }
  ul { font-size: .78rem; margin: 6px 0 0 18px; }
"""


def _page(title: str, meta: str, body: str) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{esc(title)}</title>"
            f"<style>{_CSS}</style></head><body><h1>{esc(title)}</h1><div class='meta'>{meta}</div>"
            f"<div class='grid'>{body}</div></body></html>")


def _panel(title: str, b64: str, note: str = "", wide: bool = False, mime: str = "jpeg") -> str:
    return (f"<div class='panel{' wide' if wide else ''}'><h2>{title}</h2><div class='inner'>"
            f"<img src='data:image/{mime};base64,{b64}'>"
            + (f"<p class='note'>{note}</p>" if note else "") + "</div></div>")


def _b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _font(size: int = 12) -> ImageFont.ImageFont:
    for path in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/TTF/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf",
                 "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _asciify(s: str) -> str:
    return (s.replace("—", "-").replace("–", "-").replace("‘", "'").replace("’", "'")
             .replace("“", '"').replace("”", '"').replace("…", "..."))


def _apply_frac(img: Image.Image, frac: list[float], pad: float = 0.0) -> Image.Image:
    w, h = img.size
    x1, y1 = int(max(0.0, frac[0] - pad) * w), int(max(0.0, frac[1] - pad) * h)
    x2, y2 = int(min(1.0, frac[2] + pad) * w), int(min(1.0, frac[3] + pad) * h)
    return img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))


def draw_boxes(img: Image.Image, units: list[dict], pad: float = 0.0,
               refined: list[list[float] | None] | None = None) -> Image.Image:
    """Overlay each unit's crop box (and optionally its CV-refined box)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _font(12)
    w, h = img.size
    for i, unit in enumerate(units):
        frac = unit.get("crop_frac")
        if not frac or len(frac) != 4:
            continue
        x1, y1 = int(max(0.0, frac[0] - pad) * w), int(max(0.0, frac[1] - pad) * h)
        x2, y2 = int(min(1.0, frac[2] + pad) * w), int(min(1.0, frac[3] + pad) * h)
        color = "#e08030" if refined is not None else _CROP_COLORS[i % len(_CROP_COLORS)]
        for t in range(3):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
        if refined is not None and i < len(refined) and refined[i]:
            rf = refined[i]
            rx1, ry1, rx2, ry2 = int(rf[0] * w), int(rf[1] * h), int(rf[2] * w), int(rf[3] * h)
            for t in range(3):
                draw.rectangle([rx1 - t, ry1 - t, rx2 + t, ry2 + t], outline="#40c060")
        label = _asciify(unit.get("unit_type") or f"unit {i + 1}")
        if unit.get("floor_label"):
            label += f" - {_asciify(unit['floor_label'])}"
        draw.rectangle([x1, y1 - 18, x1 + len(label) * 7 + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - 16), label, fill="white", font=font)
    return out


def _write_html(html: str, out: Path | None, prefix: str, open_browser: bool) -> Path:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        path = out
    else:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, prefix=prefix, encoding="utf-8")
        tmp.write(html)
        tmp.close()
        path = Path(tmp.name)
    if open_browser:
        webbrowser.open(path.as_uri())
    return path


def _extract(client: LLMClient, model: str, img: Image.Image, max_side: int) -> tuple[dict, dict, int]:
    img_bytes, media_type = prepare_pil(img, max_side=max_side)
    record, usage = client.extract_tool(img_bytes, media_type, model, load_prompt("extract"),
                                        load_extraction_schema())
    return record, usage, len(img_bytes)


def try_one(image: Path, client: LLMClient, model: str, *, crop_pad: float = 0.02,
            max_side: int = 3000, out: Path | None = None, open_browser: bool = True) -> tuple[Path, dict]:
    with Image.open(image) as im:
        original = im.convert("RGB")
    record, usage, nbytes = _extract(client, model, original, max_side)
    units = record.get("units") or []
    body = _panel("Original + crop boxes", _b64(draw_boxes(original, units)), wide=True)
    for i, u in enumerate(units):
        heading = (u.get("unit_type") or f"unit {i + 1}") + (f" — {u['floor_label']}" if u.get("floor_label") else "")
        meta = " · ".join(filter(None, [f"{u['bedrooms']}BR" if u.get("bedrooms") is not None else "",
                                        f"{u['bathrooms']}BA" if u.get("bathrooms") is not None else "",
                                        f"{u['total_area_sqm']} sqm" if u.get("total_area_sqm") else "",
                                        f"facing {u['facing']}" if u.get("facing") not in (None, "unknown") else ""]))
        rooms = "".join(f"<li>{esc(r.get('label', ''))}" + (f" ({r['area_sqm']} sqm)" if r.get("area_sqm") else "") + "</li>"
                        for r in u.get("rooms") or []) or "<li style='color:#888'>none extracted</li>"
        crop = _apply_frac(original, u["crop_frac"], crop_pad) if u.get("crop_frac") else original
        body += f"<div class='sep'>── {esc(heading)} ──</div>"
        body += _panel("Crop", _b64(crop), f"{esc(meta)}<ul>{rooms}</ul>")
    body += ("<div class='panel'><h2>Raw extraction JSON</h2><div class='inner'><pre>"
             f"{esc(json.dumps(record, indent=2, ensure_ascii=False))}</pre></div></div>")
    meta_line = (f"SHA256: {sha256_file(image)} | Model: {esc(model)} | Class: {esc(str(record.get('classification')))}"
                 f" | Confidence: {record.get('confidence')} | Units: {len(units)} | sent {nbytes:,} bytes"
                 f" | tokens {usage.get('input_tokens')}/{usage.get('output_tokens')}")
    html = _page(f"try-one — {image.name}", meta_line, body)
    return _write_html(html, out, f"fpextract_try_one_{image.stem}_", open_browser), record


def try_crop(image: Path, client: LLMClient, model: str, *, crop_pad: float = 0.02,
             snap_margin: float = 0.015, max_side: int = 3000, out: Path | None = None,
             open_browser: bool = True) -> tuple[Path, dict, list]:
    from fpextract.cv.crop import refine_crop

    with Image.open(image) as im:
        original = im.convert("RGB")
    record, usage, _ = _extract(client, model, original, max_side)
    units = record.get("units") or []
    refined: list[list[float] | None] = []
    for u in units:
        frac = u.get("crop_frac")
        refined.append(refine_crop(original, frac, pad=crop_pad, margin_frac=snap_margin)
                       if frac and len(frac) == 4 else None)
    body = _panel("Original — both crop boxes overlaid", _b64(draw_boxes(original, units, crop_pad, refined)), wide=True)
    for i, u in enumerate(units):
        frac = u.get("crop_frac")
        if not frac:
            continue
        heading = (u.get("unit_type") or f"unit {i + 1}") + (f" — {u['floor_label']}" if u.get("floor_label") else "")
        llm_crop = _apply_frac(original, frac, crop_pad)
        body += f"<div class='sep'>── {esc(heading)} ──</div>"
        body += _panel(f"<span class='llm'>LLM crop</span> (+{crop_pad * 100:.0f}% pad)", _b64(llm_crop),
                       f"{llm_crop.size[0]}×{llm_crop.size[1]} px [{', '.join(f'{v:.3f}' for v in frac)}]")
        rf = refined[i]
        if rf:
            cv_crop = _apply_frac(original, rf)
            body += _panel("<span class='cv'>CV-refined crop</span>", _b64(cv_crop),
                           f"{cv_crop.size[0]}×{cv_crop.size[1]} px [{', '.join(f'{v:.3f}' for v in rf)}]")
    body += ("<div class='panel'><h2>Raw extraction JSON</h2><div class='inner'><pre>"
             f"{esc(json.dumps(record, indent=2, ensure_ascii=False))}</pre></div></div>")
    meta_line = (f"Model: {esc(model)} | Class: {esc(str(record.get('classification')))} | Units: {len(units)}"
                 f" | <span class='llm'>■ orange = LLM + pad</span> <span class='cv'>■ green = CV-refined</span>"
                 f" | tokens {usage.get('input_tokens')}/{usage.get('output_tokens')}")
    html = _page(f"try-crop — {image.name}", meta_line, body)
    return _write_html(html, out, f"fpextract_try_crop_{image.stem}_", open_browser), record, refined


def try_norm(image: Path, *, crop: list[float] | None = None, crop_pad: float = 0.02,
             bg_kernel_frac: float = 0.07, tiled_watermark: bool = False,
             slbr_weights: Path | None = None, out_image: Path | None = None,
             out: Path | None = None, open_browser: bool = True) -> Path:
    from fpextract.cv import require_cv2
    from fpextract.cv.normalize import normalize_background

    require_cv2()
    import cv2
    import numpy as np

    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image: {image}")

    def png(arr) -> str:
        ok, buf = cv2.imencode(".png", arr)
        return base64.standard_b64encode(buf.tobytes()).decode("ascii")

    body = _panel("1 · Original", png(img), f"{img.shape[1]}×{img.shape[0]} px", mime="png")
    if crop:
        x1f, y1f = max(0.0, crop[0] - crop_pad), max(0.0, crop[1] - crop_pad)
        x2f, y2f = min(1.0, crop[2] + crop_pad), min(1.0, crop[3] + crop_pad)
        h0, w0 = img.shape[:2]
        img = img[int(y1f * h0):int(y2f * h0), int(x1f * w0):int(x2f * w0)]
        body += _panel("2 · Cropped", png(img), f"[{x1f:.3f}, {y1f:.3f}, {x2f:.3f}, {y2f:.3f}] → {img.shape[1]}×{img.shape[0]} px", mime="png")
    result = normalize_background(img, bg_kernel_frac=bg_kernel_frac)
    body += _panel("3 · Background normalised", png(result),
                   "Black-hat per channel → 255 − blackhat; background becomes white regardless of tint.", mime="png")
    if tiled_watermark:
        from fpextract.cv.dewatermark import remove_tiled_watermark
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        clean_rgb, info = remove_tiled_watermark(np.ascontiguousarray(rgb))
        result = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)
        body += _panel("4 · Tiled watermark removed", png(result), esc(json.dumps(info)), mime="png")
    if slbr_weights:
        from fpextract.cv.slbr import Dewatermarker
        result = Dewatermarker(slbr_weights)(result)
        body += _panel("5 · Dewatermarked (SLBR mask + TELEA)", png(result), mime="png")
    if out_image:
        out_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_image), result)
    html = _page(f"try-norm — {image.name}", "", body)
    return _write_html(html, out, f"fpextract_try_norm_{image.stem}_", open_browser)
