"""Targeted, idempotent normalisations for aggregated ``projects.jsonl``.

Every rule exists because the profiler flagged it on real data:

* R7 — recover ``unit_type`` from the source filename when the model emitted
  a room word or a generic descriptor and every source filename agrees on a
  code (``1-Bedroom-a4-b1.jpg`` → ``a4b1``).
* R1 — drop plan types whose ``unit_type`` is a room word.
* R2 — dedupe replicated stack-chart records (same unit type and specs,
  different ``floor_label``); keep the highest-confidence copy.
* R3 — null ``bedrooms`` when it is 0 on a non-studio unit type.
* R4 — null ``total_area_sqm`` when it is 0 or more than 2× outside the
  plausible band for the bedroom count.
* R5 — tag generic-descriptor unit types with ``unit_type_is_generic``.
* R6 — coerce non-cardinal ``facing`` values to ``"unknown"``.

Running the cleaner twice yields identical output.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from fpextract.classify import derive_unit_code_from_filename
from fpextract.dq.profile import (
    AREA_BANDS_SQM,
    CARDINAL_FACINGS,
    is_generic_descriptor,
    is_room_word,
    md_table,
)
from fpextract.sink.jsonl import iter_jsonl

AREA_OUTLIER_FACTOR = 2.0

_GENERIC_TRIGGER_RE = re.compile(
    r"^(?:\d+\s*-?\s*bedroom|\d+\s*-?\s*br\b|studio\b|loft\b|penthouse\b|ph\b|duplex\b)", re.I)

RULES = ("r7_filename_recovered", "r1_dropped", "r2_replicated_dedup", "r3_zero_bed_nullified",
         "r4_area_nullified", "r5_generic_tagged", "r6_facing_normalised")


def new_stats() -> dict[str, list[dict]]:
    return {k: [] for k in RULES}


def needs_filename_recovery(unit_type: str | None) -> bool:
    if not unit_type or not unit_type.strip():
        return True
    s = unit_type.strip()
    return is_room_word(s) or bool(_GENERIC_TRIGGER_RE.match(s))


def _basenames(p: dict) -> list[str]:
    if p.get("source_image_basenames"):
        return list(p["source_image_basenames"])
    return [Path(s.get("image_path") or "").name for s in (p.get("sources") or []) if s.get("image_path")]


def filename_code_consensus(p: dict) -> str | None:
    """All source filenames must derive the same code, otherwise abstain."""
    codes = []
    for bn in _basenames(p):
        c = derive_unit_code_from_filename(Path(bn).stem)
        if c is None:
            return None
        codes.append(c)
    if not codes or len(set(codes)) > 1:
        return None
    return codes[0]


def area_is_egregious(bedrooms: int | None, area: float | None) -> bool:
    if area is None:
        return False
    if area <= 0:
        return True
    if not isinstance(bedrooms, int):
        return False
    band = AREA_BANDS_SQM.get(min(bedrooms, 6))
    return bool(band) and (area < band[0] / AREA_OUTLIER_FACTOR or area > band[1] * AREA_OUTLIER_FACTOR)


def replicated_groups(plans: list[dict]) -> dict[str, list[int]]:
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for i, p in enumerate(plans):
        by_key[(p.get("unit_type"), p.get("bedrooms"), p.get("bathrooms"),
                p.get("total_area_sqm"), p.get("balcony_sqm"))].append(i)
    out: dict[str, list[int]] = {}
    for key, idxs in by_key.items():
        if len(idxs) > 1 and len({plans[i].get("floor_label") for i in idxs}) > 1:
            out[json.dumps(key, default=str)] = idxs
    return out


def clean_record(rec: dict, stats: dict[str, list[dict]]) -> dict:
    slug = rec.get("project_slug")
    plans = [dict(p) for p in (rec.get("plan_types") or [])]
    drop: set[int] = set()

    for p in plans:  # R7
        orig = p.get("unit_type")
        if not needs_filename_recovery(orig):
            continue
        code = filename_code_consensus(p)
        if code:
            p["unit_type_original"] = orig
            p["unit_type"] = code
            p["unit_type_source"] = "filename"
            stats["r7_filename_recovered"].append({"slug": slug, "was": orig, "recovered": code})

    for i, p in enumerate(plans):  # R1
        ut = (p.get("unit_type") or "").strip()
        if ut and is_room_word(ut):
            drop.add(i)
            stats["r1_dropped"].append({"slug": slug, "unit_type": ut, "reason": "room-word unit_type"})

    for _, idxs in replicated_groups(plans).items():  # R2
        live = [i for i in idxs if i not in drop]
        if len(live) <= 1:
            continue
        winner = max(live, key=lambda i: plans[i].get("confidence") or 0)
        losers = [i for i in live if i != winner]
        drop.update(losers)
        plans[winner]["floor_label"] = None
        shas = list(plans[winner].get("source_image_sha256s") or [])
        bns = list(plans[winner].get("source_image_basenames") or [])
        for j in losers:
            for s in plans[j].get("source_image_sha256s") or []:
                if s not in shas:
                    shas.append(s)
            for b in plans[j].get("source_image_basenames") or []:
                if b not in bns:
                    bns.append(b)
        if shas:
            plans[winner]["source_image_sha256s"] = shas
        if bns:
            plans[winner]["source_image_basenames"] = bns
        stats["r2_replicated_dedup"].append({"slug": slug, "unit_type": plans[winner].get("unit_type"),
                                             "n_dropped": len(losers)})

    cleaned: list[dict] = []
    for i, p in enumerate(plans):
        if i in drop:
            continue
        ut = (p.get("unit_type") or "").strip()
        bed, area, face = p.get("bedrooms"), p.get("total_area_sqm"), p.get("facing")
        if bed == 0 and ut and not is_generic_descriptor(ut) and not is_room_word(ut):  # R3
            p["bedrooms"] = None
            stats["r3_zero_bed_nullified"].append({"slug": slug, "unit_type": ut})
        if area_is_egregious(bed, area):  # R4
            p["total_area_sqm"] = None
            stats["r4_area_nullified"].append({"slug": slug, "unit_type": ut, "bedrooms": bed, "area_was": area})
        if ut and is_generic_descriptor(ut):  # R5
            p["unit_type_is_generic"] = True
            stats["r5_generic_tagged"].append({"slug": slug, "unit_type": ut})
        if face is not None and str(face).strip().lower() not in CARDINAL_FACINGS | {"unknown", ""}:  # R6
            p["facing"] = "unknown"
            stats["r6_facing_normalised"].append({"slug": slug, "unit_type": ut, "was": face})
        cleaned.append(p)
    return {**rec, "plan_types": cleaned}


def render_report(stats: dict[str, list[dict]], total_in: int, total_out: int) -> str:
    out = ["# Floor-plan data quality — cleaning report\n",
           "Rows touched by each rule. Output is a new file; the input is never modified.\n",
           "\n## Rule summary\n",
           md_table(["rule", "rows touched"], [
               ["R7 — recover unit_type from filename", len(stats["r7_filename_recovered"])],
               ["R1 — drop room-word unit_types", len(stats["r1_dropped"])],
               ["R2 — dedupe replicated stack records", len(stats["r2_replicated_dedup"])],
               ["R3 — nullify zero-bedroom counts", len(stats["r3_zero_bed_nullified"])],
               ["R4 — nullify egregious areas", len(stats["r4_area_nullified"])],
               ["R5 — tag generic-descriptor types", len(stats["r5_generic_tagged"])],
               ["R6 — normalise non-cardinal facing", len(stats["r6_facing_normalised"])]]),
           f"\n_plan_types in input:_ **{total_in}** → _output:_ **{total_out}** (net {total_out - total_in})\n"]

    def section(title: str, rows: list[dict], cols: list[str]) -> None:
        out.append(f"\n## {title}\n")
        if not rows:
            out.append("_None_")
            return
        out.append(f"Total: **{len(rows)}**. Showing first 30:\n")
        out.append(md_table(cols, [[r.get(c) for c in cols] for r in rows[:30]]))

    section("R7 — unit_type recovered from filename", stats["r7_filename_recovered"], ["slug", "was", "recovered"])
    section("R1 — dropped room-word unit_types", stats["r1_dropped"], ["slug", "unit_type", "reason"])
    section("R2 — replicated stack records", stats["r2_replicated_dedup"], ["slug", "unit_type", "n_dropped"])
    section("R3 — bedrooms 0 → null", stats["r3_zero_bed_nullified"], ["slug", "unit_type"])
    section("R4 — total_area_sqm nullified", stats["r4_area_nullified"], ["slug", "unit_type", "bedrooms", "area_was"])
    section("R5 — generic-descriptor unit_types tagged", stats["r5_generic_tagged"], ["slug", "unit_type"])
    section("R6 — facing normalised", stats["r6_facing_normalised"], ["slug", "unit_type", "was"])
    return "\n".join(out)


def run(input_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    stats = new_stats()
    n_in = n_out = n_rec = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for rec in iter_jsonl(input_path):
            n_rec += 1
            n_in += len(rec.get("plan_types") or [])
            cleaned = clean_record(rec, stats)
            n_out += len(cleaned.get("plan_types") or [])
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
    report_path.write_text(render_report(stats, n_in, n_out), encoding="utf-8")
    return {"records": n_rec, "plan_types_in": n_in, "plan_types_out": n_out,
            **{k: len(v) for k, v in stats.items()}}
