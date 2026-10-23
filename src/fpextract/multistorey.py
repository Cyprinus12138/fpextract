"""Collapse per-(image, unit) rows into per-(project, unit_type) *physical
unit* records, detecting multi-storey units (duplex, penthouse, maisonette).

Detection: a record's storey signal is the title's ``storey_label`` or,
failing that, the picture's ``floor_label`` when it contains a known
storey keyword (block / tower / stack labels are *not* storeys). Two or more
distinct signals under one ``per_unit_plan`` unit type → multi-storey:
measurements are summed across storeys and ``storeys[]`` keeps the per-level
breakdown. ``floor_plate`` units are never collapsed across storeys.

Each collapsed unit also carries a ``confidence``: the best contributing
image confidence, plus a small agreement bonus, minus a penalty per field
the sources disagree on.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import median
from typing import Any

_STOREY_RANK_MAP = [
    (("basement", "b1", "bsmt"), -10),
    (("ground", "1st storey", "lower floor", "lower level", "lower storey", "lower", "level 1"), 1),
    (("2nd storey", "upper floor", "upper level", "upper storey", "upper", "level 2"), 2),
    (("3rd storey", "level 3"), 3),
    (("4th storey", "level 4"), 4),
    (("5th storey", "level 5"), 5),
    (("6th storey", "level 6"), 6),
    (("attic", "loft", "rooftop", "roof terrace"), 100),
]

_STOREY_KEYWORD_RE = re.compile(
    r"(?ix)\b(basement | ground(\s+(floor|storey|level))? | attic | loft | rooftop | roof\s+terrace |"
    r" \d+(st|nd|rd|th)?\s*storey | \d+(st|nd|rd|th)?\s*floor | level\s*\d+ |"
    r" level\s*(one|two|three|four|five|six|seven|eight|nine|ten) | \d+(st|nd|rd|th)?\s*level |"
    r" lower(\s+(floor|storey|level))? | upper(\s+(floor|storey|level))?)\b")

_DISAGREEMENT_FIELDS = ("bedrooms", "bathrooms", "total_area_sqm", "balcony_sqm", "facing")


def storey_rank(label: str | None) -> int:
    if not label:
        return 99999
    s = label.strip().lower()
    for keys, rank in _STOREY_RANK_MAP:
        if any(k in s for k in keys):
            return rank
    if s.isdigit():          # stack-chart row numbers are not storeys of one unit
        return 1_000_000 + int(s)
    return 50_000


def storey_signal(record: dict) -> str | None:
    s = record.get("_title_storey_label")
    if isinstance(s, str) and s.strip():
        return s.strip()
    s = record.get("floor_label")
    if isinstance(s, str) and s.strip() and _STOREY_KEYWORD_RE.search(s.strip()):
        return s.strip()
    return None


def _hashable(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(w)) for k, w in v.items()))
    return v


def _mode_or_first(values: list[Any]) -> Any:
    nonnull = [v for v in values if v is not None and v != ""]
    if not nonnull:
        return None
    top, _ = Counter(_hashable(v) for v in nonnull).most_common(1)[0]
    for v in nonnull:
        if _hashable(v) == top:
            return v
    return nonnull[0]


def _median_or_none(values: list[Any]) -> float | int | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return None
    m = median(nums)
    if all(isinstance(v, int) for v in nums) and float(m).is_integer():
        return int(m)
    return round(float(m), 2)


def _flatten_unique(lists: list[list[dict]]) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for ld in lists:
        for d in ld or []:
            sig = _hashable(d)
            if sig not in seen:
                seen.add(sig)
                out.append(d)
    return out


def _sum_optional(values: Iterable[Any]) -> Any:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return None
    total = sum(nums)
    return int(total) if all(isinstance(v, int) for v in nums) else round(float(total), 2)


def consensus_merge(records: list[dict]) -> dict:
    """Combine N records of the same storey: mode for counts and facing,
    median for areas, union for rooms and windows."""
    if not records:
        return {}
    return {
        "bedrooms": _mode_or_first([r.get("bedrooms") for r in records]),
        "bathrooms": _mode_or_first([r.get("bathrooms") for r in records]),
        "total_area_sqm": _median_or_none([r.get("total_area_sqm") for r in records]),
        "balcony_sqm": _median_or_none([r.get("balcony_sqm") for r in records]),
        "facing": _mode_or_first([r.get("facing") for r in records]),
        "rooms": _flatten_unique([r.get("rooms") for r in records]),
        "windows": _flatten_unique([r.get("windows") for r in records]),
        "n_source_records": len(records),
    }


def unit_confidence(records: list[dict]) -> tuple[float, list[str]]:
    """``(confidence, field_disagreements)`` across the contributing records."""
    confs = [r.get("_image_confidence") for r in records]
    confs = [float(c) for c in confs if isinstance(c, (int, float))] or [1.0]
    disagreements: list[str] = []
    for f in _DISAGREEMENT_FIELDS:
        seen = set()
        for r in records:
            v = r.get(f)
            if f == "facing" and isinstance(v, str) and v.lower() == "unknown":
                v = None
            if v is not None:
                seen.add(_hashable(v))
        if len(seen) > 1:
            disagreements.append(f)
    bonus = min(0.05 * (len(records) - 1), 0.10)
    conf = max(0.0, min(1.0, max(confs) + bonus - 0.10 * len(disagreements)))
    return round(conf, 3), disagreements


def collapse_unit(records: list[dict]) -> dict:
    """All enriched rows for one ``(slug, unit_type)`` → one physical unit."""
    cls = _mode_or_first([r.get("_classification") for r in records]) or "per_unit_plan"
    pt = _mode_or_first([r.get("_property_type") for r in records]) or "unknown"
    sha_set: set[str] = set()
    basenames: list[str] = []
    for r in records:
        sh = r.get("_image_sha256")
        if sh and sh not in sha_set:
            sha_set.add(sh)
            ip = r.get("_image_path") or ""
            if ip:
                basenames.append(ip.replace("\\", "/").rsplit("/", 1)[-1])
    conf, disagreements = unit_confidence(records)
    common = {
        "project_slug": records[0].get("project_slug") or "",
        "unit_type": records[0].get("unit_type") or "",
        "property_type": pt,
        "classification": cls,
        "confidence": conf,
        "field_disagreements": disagreements,
        "n_source_records": len(records),
        "source_image_sha256s": sorted(sha_set),
        "source_image_basenames": basenames,
        "match_methods": sorted({r.get("_match_method") for r in records if r.get("_match_method")}),
    }
    if cls == "floor_plate":
        return {**common, "n_storeys": 1, **consensus_merge(records)}

    by_storey: dict[str | None, list[dict]] = defaultdict(list)
    for r in records:
        by_storey[storey_signal(r)].append(r)
    if any(k is not None for k in by_storey) and None in by_storey:
        biggest = max((k for k in by_storey if k is not None), key=lambda k: len(by_storey[k]))
        by_storey[biggest].extend(by_storey.pop(None))
    if len(by_storey) <= 1:
        return {**common, "n_storeys": 1, **consensus_merge(records)}

    storeys = [{"storey_label": label, **consensus_merge(rs)}
               for label, rs in sorted(by_storey.items(), key=lambda kv: storey_rank(kv[0]))]
    return {
        **common,
        "n_storeys": len(storeys),
        "storeys": storeys,
        "bedrooms": _sum_optional(s.get("bedrooms") for s in storeys),
        "bathrooms": _sum_optional(s.get("bathrooms") for s in storeys),
        "total_area_sqm": _sum_optional(s.get("total_area_sqm") for s in storeys),
        "balcony_sqm": _sum_optional(s.get("balcony_sqm") for s in storeys),
        "facing": max(storeys, key=lambda s: s.get("total_area_sqm") or 0).get("facing"),
        "rooms": [r for s in storeys for r in (s.get("rooms") or [])],
        "windows": [w for s in storeys for w in (s.get("windows") or [])],
    }


def collapse_project(enriched: list[dict]) -> list[dict]:
    """Group one project's enriched rows by unit type and collapse each."""
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for r in enriched:
        ut = (r.get("unit_type") or "").strip()
        if ut:
            by_unit[ut].append(r)
    return [collapse_unit(rs) for _, rs in sorted(by_unit.items())]
