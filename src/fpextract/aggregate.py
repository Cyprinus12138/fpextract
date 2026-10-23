"""Stage 2 — one record per project, with provenance and confidence.

Chain (all local, no API spend, regenerable at any time)::

    extractions.jsonl ─┬─► enrich (optional sheet-title sidecar)
                       │        │
                       │        ▼
                       │   collapse per (project, unit_type)   → plan_types[]
                       │
                       └─► merge unit_number_map across images → unit_number_map,
                                                                  by_unit_type,
                                                                  unit_number_evidence,
                                                                  conflicts

Per-project record (abridged)::

    {"project_slug", "project_title", "project_url", "n_source_images",
     "n_physical_units", "n_multi_storey_units",
     "property_types_seen", "dominant_property_type", "property_type_source",
     "confidence", "plan_types": [...], "unit_number_map": {...},
     "by_unit_type": {...}, "unit_number_evidence": {...}, "conflicts": [...]}
"""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter, defaultdict
from pathlib import Path

from fpextract.enrich import enrich_all, load_sidecar
from fpextract.inputs import ProjectMeta, load_project_meta
from fpextract.multistorey import collapse_project
from fpextract.sink.jsonl import load_extractions

# ---------------------------------------------------------------------------
# Project-level property_type inference
# ---------------------------------------------------------------------------
# Per-record votes are unreliable on landed-house projects whose sheets are
# titled with bare street addresses. Two project-level signals override them:
# (a) keywords in the project's free-text description (``description`` in the
# manifest), (b) address-shaped unit types.

_DESCRIPTION_KEYWORDS = [
    (re.compile(r"\b(semi[\s-]?detached|bungalow|terrace|cluster\s+house|townhouse|"
                r"detached\s+house|landed)\b", re.I), "landed_house"),
    (re.compile(r"\b(warehouse|factory|industrial|production\s+unit|"
                r"b1\s+industrial|b2\s+industrial)\b", re.I), "industrial"),
    (re.compile(r"\b(shop\s+unit|retail\s+unit|commercial\s+unit|office\s+unit|f&b\s+unit)\b",
                re.I), "commercial"),
]
_LANDED_HOUSE_TITLE_RE = re.compile(r"\bhouse\s+\w+", re.I)
_ADDRESS_RE = re.compile(
    r"^\s*\d+\s*[A-Z]?\s*(?:&|and|to)?\s*\d*\s*[A-Z]?\s+[A-Z][\w\s'-]*?\s+"
    r"(road|drive|lane|avenue|place|street|park|crescent|walk|way|close|hill|view|grove|rise|"
    r"terrace|gardens?|heights?|mews|lodge|loop|link|bend|circle|circuit|court|gate|green|edge|"
    r"quay|ridge|valley|vista|pt\.?|point|expressway)\b", re.IGNORECASE)


def infer_property_type_from_text(text: str) -> str | None:
    if not text:
        return None
    for pat, pt in _DESCRIPTION_KEYWORDS:
        if pat.search(text):
            return pt
    return None


def looks_like_address(unit_type: str) -> bool:
    return bool(unit_type) and bool(_ADDRESS_RE.match(unit_type) or _LANDED_HOUSE_TITLE_RE.match(unit_type))


def infer_project_property_type(meta: ProjectMeta | None, units: list[dict]) -> str | None:
    """Forced project-level property type when confident, else ``None``."""
    if meta is not None:
        if meta.property_type:
            return meta.property_type
        hit = infer_property_type_from_text(meta.description)
        if hit:
            return hit
    per_unit = [u for u in units if u.get("classification") == "per_unit_plan"]
    if not per_unit:
        return None
    n_addr = sum(1 for u in per_unit if looks_like_address(u.get("unit_type") or ""))
    return "landed_house" if n_addr / len(per_unit) >= 0.5 else None


# ---------------------------------------------------------------------------
# unit_number_map aggregation
# ---------------------------------------------------------------------------

def _llm_confidence(rec: dict) -> float:
    c = rec.get("confidence")
    return float(c) if isinstance(c, (int, float)) else 1.0


def merge_unit_number_map(records: list[dict]) -> tuple[dict[str, str], dict[str, dict],
                                                        dict[str, list[str]], list[dict]]:
    """Union every image's ``unit_number_map``; when images disagree on a unit
    number's type the most-voted candidate wins (tie-break: highest image
    confidence). Returns ``(flat_map, evidence, by_unit_type, conflicts)``."""
    votes: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        unmap = r.get("unit_number_map") or {}
        if not isinstance(unmap, dict):
            continue
        for un, ut in unmap.items():
            if un and ut:
                votes[un][ut].append({"image_path": r.get("image_path"),
                                      "llm_confidence": _llm_confidence(r)})
    flat_map: dict[str, str] = {}
    evidence: dict[str, dict] = {}
    conflicts: list[dict] = []
    for un, type_votes in votes.items():
        ranked = sorted(type_votes.items(),
                        key=lambda kv: (-len(kv[1]), -max(s["llm_confidence"] for s in kv[1])))
        winning, winning_sources = ranked[0]
        flat_map[un] = winning
        total = sum(len(s) for s in type_votes.values())
        confidence = round(len(winning_sources) / total * max(s["llm_confidence"] for s in winning_sources), 3)
        conflicting = [{"unit_type": t, "source_count": len(s)} for t, s in ranked[1:]]
        evidence[un] = {"unit_type": winning, "confidence": confidence,
                        "source_count": total, "conflicting_candidates": conflicting}
        if conflicting:
            conflicts.append({"unit_number": un, "winning": winning,
                              "candidates": [{"unit_type": t, "source_count": len(s)} for t, s in ranked]})
    by_unit_type: dict[str, list[str]] = defaultdict(list)
    for un, ut in flat_map.items():
        by_unit_type[ut].append(un)
    return flat_map, evidence, {ut: sorted(v) for ut, v in sorted(by_unit_type.items())}, conflicts


# ---------------------------------------------------------------------------
# Project assembly
# ---------------------------------------------------------------------------

def project_property_type(units: list[dict], inferred: str | None) -> tuple[list[str], str]:
    counter: Counter = Counter(u.get("property_type") or "unknown" for u in units
                               if u.get("classification") == "per_unit_plan")
    seen = sorted(counter)
    if inferred:
        return sorted(set(seen) | {inferred}), inferred
    if not counter:
        return [], "unknown"
    return seen, counter.most_common(1)[0][0]


def shape_plan_type(unit: dict) -> dict:
    return {
        "unit_type": unit.get("unit_type"),
        "property_type": unit.get("property_type"),
        "classification": unit.get("classification"),
        "n_storeys": unit.get("n_storeys"),
        "bedrooms": unit.get("bedrooms"),
        "bathrooms": unit.get("bathrooms"),
        "total_area_sqm": unit.get("total_area_sqm"),
        "balcony_sqm": unit.get("balcony_sqm"),
        "facing": unit.get("facing"),
        "rooms": unit.get("rooms") or [],
        "windows": unit.get("windows") or [],
        "storeys": unit.get("storeys") if (unit.get("n_storeys") or 1) > 1 else None,
        "confidence": unit.get("confidence"),
        "field_disagreements": unit.get("field_disagreements") or [],
        "n_source_records": unit.get("n_source_records"),
        "source_image_sha256s": unit.get("source_image_sha256s") or [],
        "source_image_basenames": unit.get("source_image_basenames") or [],
        "match_methods": unit.get("match_methods") or [],
    }


def project_confidence(plan_types: list[dict], evidence: dict[str, dict]) -> float | None:
    confs = [p["confidence"] for p in plan_types if isinstance(p.get("confidence"), (int, float))]
    confs += [e["confidence"] for e in evidence.values() if isinstance(e.get("confidence"), (int, float))]
    return round(sum(confs) / len(confs), 3) if confs else None


def build_project_record(slug: str, raw_records: list[dict], collapsed_units: list[dict],
                         meta: ProjectMeta | None) -> dict:
    inferred = infer_project_property_type(meta, collapsed_units)
    if inferred:
        for u in collapsed_units:
            if u.get("classification") in ("per_unit_plan", "floor_plate"):
                u["property_type"] = inferred
    plan_types = [shape_plan_type(u) for u in collapsed_units]
    cls_priority = {"per_unit_plan": 0, "floor_plate": 1}
    plan_types.sort(key=lambda p: (cls_priority.get(p.get("classification"), 9),
                                   -(p.get("total_area_sqm") or 0), p.get("unit_type") or ""))
    flat_map, evidence, by_unit_type, conflicts = merge_unit_number_map(raw_records)
    seen, dominant = project_property_type(collapsed_units, inferred)
    return {
        "project_slug": slug,
        "project_title": (meta.title if meta else "") or slug,
        "project_url": meta.url if meta else "",
        "n_source_images": len(raw_records),
        "n_physical_units": len(collapsed_units),
        "n_multi_storey_units": sum(1 for u in collapsed_units if (u.get("n_storeys") or 1) > 1),
        "property_types_seen": seen,
        "dominant_property_type": dominant,
        "property_type_source": "project_inferred" if inferred else "llm_voted",
        "aggregated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "confidence": project_confidence(plan_types, evidence),
        "plan_types": plan_types,
        "unit_number_map": flat_map,
        "by_unit_type": by_unit_type,
        "unit_number_evidence": evidence,
        "conflicts": conflicts,
    }


def aggregate(extractions: Path, *, input_source: Path | None = None, sidecar: Path | None = None,
              projects_only: set[str] | None = None) -> tuple[list[dict], dict]:
    """Run the whole stage. Returns ``(project_records, stats)``."""
    by_project = load_extractions(extractions)
    meta = load_project_meta(input_source) if input_source else {}
    lookup = load_sidecar(sidecar)
    enriched, stats = enrich_all(by_project, lookup)
    out: list[dict] = []
    for slug in sorted(by_project):
        if projects_only and slug not in projects_only:
            continue
        collapsed = collapse_project(enriched.get(slug, []))
        out.append(build_project_record(slug, by_project[slug], collapsed, meta.get(slug)))
    stats["projects"] = len(out)
    stats["multi_storey_units"] = sum(r["n_multi_storey_units"] for r in out)
    stats["conflicts"] = sum(len(r["conflicts"]) for r in out)
    return out, stats
