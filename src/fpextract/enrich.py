"""Optional reconciliation of picture-extracted units with a *sheet-title
sidecar* — ground truth about each image that the picture itself may not
carry (its caption on the source page, a human-entered type code, …).

Sidecar JSONL, one row per image::

    {"project": "harbour-view", "image": "hv-a1.png",
     "title": "Type A1 — 2 Bedroom",            # optional, used for sanity rules
     "classification": "per_unit_plan",          # see CLASSIFICATIONS
     "property_type": "condo_apartment",         # free label, e.g. landed_house
     "plan_types": ["A1"],                       # unit codes the title names
     "storey_label": "Upper Floor"}              # optional

``image`` matches the extraction's ``image_path`` basename (case-insensitive).

Reconciliation rules
--------------------
1. No sidecar row → keep the picture's units verbatim, tagged ``_no_title``.
2. Title classification in :data:`DROP_CLASSIFICATIONS` → drop the image's
   units (site plans, stack charts, covers contribute nothing to plan types).
3. ``floor_plate`` → keep picture units verbatim (each cell is its own unit).
4. ``per_unit_plan`` → the title's ``plan_types`` override the picture's
   ``unit_type``: 1 title type covers every picture unit; N titles × N units
   pair in document order; N × M>N keeps the top-N by area; N × M<N cycles.

The picture's original label is always preserved in ``_picture_unit_type``.
"""
from __future__ import annotations

import re
from pathlib import Path

from fpextract.sink.jsonl import iter_jsonl

CLASSIFICATIONS = {"per_unit_plan", "floor_plate", "site_plan", "stack_chart",
                   "diagrammatic_chart", "brochure_cover", "other"}
DROP_CLASSIFICATIONS = {"site_plan", "stack_chart", "diagrammatic_chart", "brochure_cover", "other"}

# Titles whose text names a project-wide artefact. A title extractor that
# tags these per_unit_plan leaks the project name as a plan type.
_NON_UNIT_TITLE_RE = re.compile(
    r"(?ix)\b(site\s+map | site\s+plan | unit\s+distribution\s+chart | distribution\s+chart |"
    r" facilities | facilit(y|ies)\s+plan | diagrammatic | key\s+plan | location\s+plan |"
    r" swimming\s+pool | pool\s+(level|deck) | balcony\s+screen)\b")


def _slug_normalised(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def looks_like_slug_leak(plan_type: str, slug: str) -> bool:
    """The project's own name returned as a unit code."""
    return bool(plan_type) and _slug_normalised(plan_type) == _slug_normalised(slug)


def coerce_meta(meta: dict | None, slug: str) -> dict | None:
    """Defensive corrections to one sidecar row."""
    if meta is None:
        return None
    cls = meta.get("classification") or "other"
    title = meta.get("title") or ""
    plan_types = list(meta.get("plan_types") or [])
    if cls == "per_unit_plan" and _NON_UNIT_TITLE_RE.search(title):
        cls = "site_plan" if "site" in title.lower() else "other"
        plan_types = []
    plan_types = [pt for pt in plan_types if not looks_like_slug_leak(pt, slug)]
    if cls == "per_unit_plan" and not plan_types:
        cls = "other"
    return {**meta, "classification": cls, "plan_types": plan_types,
            "property_type": meta.get("property_type", "unknown") if cls == "per_unit_plan" else "unknown"}


def load_sidecar(path: Path | None) -> dict[tuple[str, str], dict]:
    """``(slug, image_basename_lower) → row``. Empty when ``path`` is None."""
    out: dict[tuple[str, str], dict] = {}
    if path is None:
        return out
    if not path.exists():
        raise SystemExit(f"sheet-title sidecar not found: {path}")
    for rec in iter_jsonl(path):
        if "error" in rec:
            continue
        slug = (rec.get("project") or rec.get("slug") or "").strip()
        image = rec.get("image") or rec.get("image_path") or ""
        if not slug or not image:
            continue
        bn = Path(image).name.lower()
        out[(slug, bn)] = {
            "classification": rec.get("classification") or "other",
            "property_type": rec.get("property_type") or "unknown",
            "plan_types": rec.get("plan_types") or [],
            "storey_label": rec.get("storey_label"),
            "title": rec.get("title") or "",
        }
    return out


def _audit(unit: dict, rec: dict, meta: dict | None, extra: dict | None = None) -> dict:
    out = dict(unit)
    out["project_slug"] = rec.get("project_slug") or ""
    out["_image_sha256"] = rec.get("image_sha256") or ""
    if rec.get("image_path"):
        out["_image_path"] = rec["image_path"]
    c = rec.get("confidence")
    out["_image_confidence"] = float(c) if isinstance(c, (int, float)) else 1.0
    if meta is not None:
        out["_classification"] = meta["classification"]
        out["_property_type"] = meta["property_type"]
        out["_title"] = meta.get("title", "")
        out["_title_storey_label"] = meta.get("storey_label")
    if extra:
        out.update(extra)
    return out


def enrich_image(rec: dict, meta: dict | None) -> list[dict]:
    """Apply the reconciliation rules to one extraction row → 0..N unit rows."""
    slug = rec.get("project_slug") or ""
    units = rec.get("units") or []
    if meta is None:
        return [_audit(u, rec, None, {"_no_title": True, "_picture_unit_type": u.get("unit_type")})
                for u in units]
    meta = coerce_meta(meta, slug)
    assert meta is not None
    cls = meta["classification"]
    if cls in DROP_CLASSIFICATIONS:
        return []
    if cls == "floor_plate":
        return [_audit(u, rec, meta, {"_picture_unit_type": u.get("unit_type")}) for u in units]

    plan_types = meta["plan_types"]
    if not plan_types:
        return [_audit(u, rec, meta, {"_picture_unit_type": u.get("unit_type"),
                                      "_match_method": "title_no_plan_types"}) for u in units]
    n_plan, n_units = len(plan_types), len(units)
    if n_plan == 1:
        if n_units == 0:
            return [_audit({"unit_type": plan_types[0]}, rec, meta,
                           {"_match_method": "title_only_no_picture_units"})]
        return [_audit(u, rec, meta, {"unit_type": plan_types[0], "_picture_unit_type": u.get("unit_type"),
                                      "_match_method": "single_plan_per_image"}) for u in units]
    if n_units == n_plan:
        return [_audit(u, rec, meta, {"unit_type": pt, "_picture_unit_type": u.get("unit_type"),
                                      "_match_method": "doc_order_NxN"})
                for pt, u in zip(plan_types, units, strict=True)]
    if n_units > n_plan:
        top = sorted(units, key=lambda u: -(u.get("total_area_sqm") or 0))[:n_plan]
        return [_audit(u, rec, meta, {"unit_type": pt, "_picture_unit_type": u.get("unit_type"),
                                      "_match_method": "area_rank_NxM_keep_top"})
                for pt, u in zip(plan_types, top, strict=True)]
    return [_audit(units[i % n_units] if n_units else {}, rec, meta,
                   {"unit_type": pt,
                    "_picture_unit_type": units[i % n_units].get("unit_type") if n_units else None,
                    "_match_method": "cycle_replicate_fallback"})
            for i, pt in enumerate(plan_types)]


def enrich_all(by_project: dict[str, list[dict]],
               sidecar: dict[tuple[str, str], dict]) -> tuple[dict[str, list[dict]], dict]:
    """Run :func:`enrich_image` over every project. Returns
    ``(enriched_by_project, stats)``."""
    out: dict[str, list[dict]] = {}
    stats: dict[str, int] = {"images": 0, "images_without_title": 0, "records": 0}
    for slug, recs in by_project.items():
        rows: list[dict] = []
        for rec in recs:
            stats["images"] += 1
            bn = Path(rec.get("image_path") or "").name.lower()
            meta = sidecar.get((slug, bn)) if sidecar else None
            if sidecar and meta is None:
                stats["images_without_title"] += 1
            rows.extend(enrich_image(rec, meta))
        stats["records"] += len(rows)
        out[slug] = rows
    return out, stats
