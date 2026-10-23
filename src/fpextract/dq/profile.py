"""Read-only data-quality profiler for aggregated ``projects.jsonl``.

Makes the long tail visible so cleaning rules can be chosen deliberately.
Sections: corpus summary, unit_type frequency (room words, generic
descriptors, phantom types), numeric distributions, cross-field invariants,
per-project anomalies, unit-number conflict leaderboard.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fpextract.sink.jsonl import iter_jsonl

ROOM_WORDS = {
    "bedroom", "master bedroom", "common bedroom", "junior bedroom",
    "bathroom", "master bathroom", "wc", "ensuite", "en-suite",
    "kitchen", "dry kitchen", "wet kitchen", "yard",
    "living", "living room", "dining", "dining room", "living/dining",
    "balcony", "private lift lobby", "lobby", "foyer",
    "study", "utility", "store", "storage", "service yard",
    "household shelter", "hs", "ac ledge", "planter",
}
GENERIC_DESCRIPTORS = {
    "studio", "1-bedroom", "1 bedroom", "2-bedroom", "2 bedroom", "3-bedroom", "3 bedroom",
    "4-bedroom", "4 bedroom", "5-bedroom", "5 bedroom",
    "1br", "2br", "3br", "4br", "5br", "1-br", "2-br", "3-br", "4-br", "5-br",
    "penthouse", "ph", "duplex", "loft",
}
NUMERIC_RE = re.compile(r"^\d+$")
NUMERIC_RANGE_RE = re.compile(r"^\d+\s*[-–—]\s*\d+$")
STOREY_LABEL_HINTS = ("ground", "1st", "2nd", "3rd", "4th", "lower", "upper", "attic",
                      "basement", "level", "storey", "floor")
CARDINAL_FACINGS = {
    "north", "south", "east", "west", "north-east", "north-west", "south-east", "south-west",
    "northeast", "northwest", "southeast", "southwest", "ne", "nw", "se", "sw", "n", "s", "e", "w",
}
# Plausible total_area_sqm by bedroom count; outside = suspicious (often sqft read as sqm).
AREA_BANDS_SQM = {0: (25, 60), 1: (30, 80), 2: (50, 130), 3: (75, 200),
                  4: (110, 350), 5: (160, 500), 6: (200, 700)}


def is_room_word(s: str) -> bool:
    return s.strip().lower() in ROOM_WORDS


def is_generic_descriptor(s: str) -> bool:
    return s.strip().lower() in GENERIC_DESCRIPTORS


def classify_floor_label(v: Any) -> str:
    if v is None or v == "":
        return "null"
    s = str(v).strip()
    if NUMERIC_RE.match(s):
        return "numeric"
    if NUMERIC_RANGE_RE.match(s):
        return "numeric_range"
    if any(h in s.lower() for h in STOREY_LABEL_HINTS):
        return "storey_label"
    return "other"


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)

    def pct(p: float) -> float:
        return s[int(round(p / 100 * (len(s) - 1)))]

    return {"n": len(s), "min": s[0], "p1": pct(1), "p25": pct(25), "p50": pct(50),
            "mean": round(statistics.mean(s), 2), "p75": pct(75), "p99": pct(99), "max": s[-1]}


def md_table(headers: list[str], rows: list) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _plan_basenames(p: dict) -> list[str]:
    if p.get("source_image_basenames"):
        return list(p["source_image_basenames"])
    return [Path(s.get("image_path") or "").name for s in (p.get("sources") or [])]


def profile(records: list[dict]) -> dict[str, Any]:
    t = {"n_projects": len(records), "n_plan_types": 0, "n_unit_numbers": 0,
         "proj_no_plans": 0, "proj_no_units": 0}
    plan_confs: list[float] = []
    proj_confs: list[float] = []
    classification_c: Counter = Counter()
    property_type_c: Counter = Counter()
    n_storeys_c: Counter = Counter()
    dominant_c: Counter = Counter()
    source_c: Counter = Counter()
    multi_storey = 0
    ut_c: Counter = Counter()
    ut_room: Counter = Counter()
    ut_generic: Counter = Counter()
    fl_classes: Counter = Counter()
    fl_top: Counter = Counter()
    facing_c: Counter = Counter()
    bedrooms: list[int] = []
    bathrooms: list[int] = []
    areas: list[float] = []
    inv = {"bath_gt_bed": [], "zero_bed": [], "area_outlier": [], "facing_bad": []}
    an = {"replicated_stacks": [], "phantom_plan_types": [], "low_conf_projects": []}
    conflict_top: list[tuple[str, int]] = []

    for rec in records:
        slug = rec.get("project_slug", "?")
        plans = rec.get("plan_types") or []
        unit_map = rec.get("unit_number_map") or {}
        by_ut_map = rec.get("by_unit_type") or {}
        if rec.get("confidence") is not None:
            proj_confs.append(rec["confidence"])
            if rec["confidence"] < 0.7:
                an["low_conf_projects"].append((slug, rec["confidence"]))
        if not plans:
            t["proj_no_plans"] += 1
        if not unit_map:
            t["proj_no_units"] += 1
        if rec.get("dominant_property_type"):
            dominant_c[rec["dominant_property_type"]] += 1
        if rec.get("property_type_source"):
            source_c[rec["property_type_source"]] += 1
        t["n_plan_types"] += len(plans)
        t["n_unit_numbers"] += len(unit_map)
        if rec.get("conflicts"):
            conflict_top.append((slug, len(rec["conflicts"])))

        grouped: dict[str, list[dict]] = defaultdict(list)
        for p in plans:
            grouped[p.get("unit_type") or ""].append(p)
        for ut, items in grouped.items():
            if len(items) > 1 and len({(p.get("bedrooms"), p.get("bathrooms"), p.get("total_area_sqm"),
                                        p.get("balcony_sqm")) for p in items}) == 1:
                an["replicated_stacks"].append((slug, ut, len(items)))

        for p in plans:
            ut = (p.get("unit_type") or "").strip()
            bed, bath, area, face = p.get("bedrooms"), p.get("bathrooms"), p.get("total_area_sqm"), p.get("facing")
            if p.get("confidence") is not None:
                plan_confs.append(p["confidence"])
            if p.get("classification"):
                classification_c[p["classification"]] += 1
            if p.get("property_type"):
                property_type_c[p["property_type"]] += 1
            nst = p.get("n_storeys") or 1
            n_storeys_c[nst] += 1
            if isinstance(nst, int) and nst > 1:
                multi_storey += 1
            if ut:
                ut_c[ut] += 1
                if is_room_word(ut):
                    ut_room[ut] += 1
                elif is_generic_descriptor(ut):
                    ut_generic[ut] += 1
            fl_classes[classify_floor_label(p.get("floor_label"))] += 1
            if p.get("floor_label"):
                fl_top[str(p["floor_label"])] += 1
            if face is not None:
                facing_c[str(face).strip().lower()] += 1
            if isinstance(bed, int):
                bedrooms.append(bed)
            if isinstance(bath, int):
                bathrooms.append(bath)
            if isinstance(area, (int, float)):
                areas.append(float(area))
            if isinstance(bed, int) and isinstance(bath, int) and bath > bed + 2:
                inv["bath_gt_bed"].append((slug, ut, bed, bath))
            if bed == 0 and ut and not is_room_word(ut) and not is_generic_descriptor(ut):
                inv["zero_bed"].append((slug, ut))
            if isinstance(bed, int) and isinstance(area, (int, float)):
                band = AREA_BANDS_SQM.get(min(bed, 6))
                if band and not (band[0] <= area <= band[1]):
                    inv["area_outlier"].append((slug, ut, bed, float(area)))
            if face is not None and str(face).strip().lower() not in CARDINAL_FACINGS | {"unknown", ""}:
                inv["facing_bad"].append((slug, ut, str(face)))
            if ut and unit_map and ut not in by_ut_map:
                an["phantom_plan_types"].append((slug, ut))

    return {
        "totals": t,
        "confidence": {"plan": percentiles(plan_confs), "project": percentiles(proj_confs)},
        "unit_type": {"top": ut_c.most_common(30), "room_words": ut_room.most_common(30),
                      "generic": ut_generic.most_common(30), "uniq": len(ut_c)},
        "floor_label": {"classes": fl_classes.most_common(), "top": fl_top.most_common(30)},
        "facing": facing_c.most_common(),
        "bedrooms": Counter(bedrooms).most_common(),
        "bathrooms": Counter(bathrooms).most_common(),
        "area_pct": percentiles(areas),
        "invariants": inv,
        "anomalies": an,
        "conflicts": {"top_projects": sorted(conflict_top, key=lambda x: -x[1])[:30],
                      "n_total": sum(c for _, c in conflict_top)},
        "v2": {"classification": classification_c.most_common(),
               "property_type": property_type_c.most_common(),
               "n_storeys": sorted(n_storeys_c.items()),
               "project_dominant_pt": dominant_c.most_common(),
               "project_pt_source": source_c.most_common(),
               "multi_storey_units": multi_storey},
    }


_PCT_KEYS = ("n", "min", "p1", "p25", "p50", "mean", "p75", "p99", "max")


def render(report: dict[str, Any], src_name: str) -> str:
    out: list[str] = ["# Floor-plan data quality report\n",
                      f"_Source: `{src_name}` — {report['totals']['n_projects']} projects_\n"]
    t = report["totals"]
    out.append("## 1. Corpus summary\n")
    out.append(md_table(["metric", "value"], [
        ["projects", t["n_projects"]], ["plan_types (total)", t["n_plan_types"]],
        ["unit_numbers (total)", t["n_unit_numbers"]],
        ["projects with no plan_types", t["proj_no_plans"]],
        ["projects with no unit_number_map", t["proj_no_units"]]]))
    out.append("\n### Confidence percentiles\n")
    out.append(md_table(["scope", *_PCT_KEYS], [
        ["per plan_type", *[report["confidence"]["plan"].get(k, "—") for k in _PCT_KEYS]],
        ["per project", *[report["confidence"]["project"].get(k, "—") for k in _PCT_KEYS]]]))

    v2 = report["v2"]
    out.append("\n## 1b. Classification and property-type signals\n")
    if v2["project_dominant_pt"]:
        out.append("\n### Project-level dominant property_type\n")
        out.append(md_table(["dominant_property_type", "projects"], v2["project_dominant_pt"]))
    if v2["project_pt_source"]:
        out.append("\n### How dominant_property_type was decided\n")
        out.append("`project_inferred` = forced by the manifest description / hint or address-shaped "
                   "unit types; `llm_voted` = mode of per-record votes.\n\n")
        out.append(md_table(["source", "projects"], v2["project_pt_source"]))
    if v2["classification"]:
        out.append("\n### plan_type classification distribution\n")
        out.append(md_table(["classification", "plan_types"], v2["classification"]))
    if v2["property_type"]:
        out.append("\n### plan_type property_type distribution\n")
        out.append(md_table(["property_type", "plan_types"], v2["property_type"]))
    if v2["n_storeys"]:
        out.append(f"\n### Multi-storey distribution\n\nTotal multi-storey units: **{v2['multi_storey_units']:,}**\n\n")
        out.append(md_table(["n_storeys", "units"], v2["n_storeys"]))

    ut = report["unit_type"]
    out.append(f"\n## 2. unit_type analysis\n\nDistinct unit_type values: **{ut['uniq']}**\n")
    out.append("\n### Top 30 unit_type values\n")
    out.append(md_table(["unit_type", "count"], ut["top"]))
    out.append("\n### unit_type values matching a room-word blocklist (likely extraction errors)\n")
    out.append(md_table(["unit_type", "count"], ut["room_words"]) if ut["room_words"] else "_None_")
    out.append("\n### unit_type values that are generic descriptors (e.g. '1-Bedroom', 'Studio')\n")
    out.append("Valid tags but useless as identifiers; they usually mean the model could not read a unit code.\n")
    out.append(md_table(["unit_type", "count"], ut["generic"]) if ut["generic"] else "_None_")

    fl = report["floor_label"]
    out.append("\n## 3. floor_label analysis\n\n### Class distribution\n")
    out.append(md_table(["class", "count"], fl["classes"]))
    out.append("\n### Top 30 floor_label values\n")
    out.append("Numeric values usually come from stack-chart confusion.\n\n")
    out.append(md_table(["floor_label", "count"], fl["top"]) if fl["top"] else "_None_")

    out.append("\n## 4. Numeric field distributions\n\n### Bedrooms\n")
    out.append(md_table(["bedrooms", "count"], sorted(report["bedrooms"])))
    out.append("\n### Bathrooms\n")
    out.append(md_table(["bathrooms", "count"], sorted(report["bathrooms"])))
    out.append("\n### total_area_sqm percentiles\n")
    ap = report["area_pct"]
    if ap:
        out.append(md_table(list(_PCT_KEYS), [[ap.get(k, "—") for k in _PCT_KEYS]]))
    out.append("\n### Facing values (top 20)\n")
    out.append(md_table(["facing", "count"], report["facing"][:20]))

    inv = report["invariants"]
    out.append("\n## 5. Cross-field invariant violations\n")
    out.append(f"\n### bathrooms > bedrooms + 2 ({len(inv['bath_gt_bed'])} rows)\n")
    if inv["bath_gt_bed"]:
        out.append(md_table(["slug", "unit_type", "bed", "bath"], inv["bath_gt_bed"][:30]))
    out.append(f"\n### bedrooms == 0 with non-room-word unit_type ({len(inv['zero_bed'])} rows)\n")
    if inv["zero_bed"]:
        out.append(md_table(["unit_type", "count"], Counter(u for _, u in inv["zero_bed"]).most_common(30)))
    out.append(f"\n### total_area_sqm outside plausible band ({len(inv['area_outlier'])} rows)\n")
    out.append("Bands: studio 25–60, 1BR 30–80, 2BR 50–130, 3BR 75–200, 4BR 110–350, 5BR 160–500, 6BR+ 200–700.\n")
    if inv["area_outlier"]:
        out.append(md_table(["slug", "unit_type", "bed", "area_sqm"], inv["area_outlier"][:30]))
    out.append(f"\n### facing not in cardinal set ({len(inv['facing_bad'])} rows)\n")
    if inv["facing_bad"]:
        out.append(md_table(["facing", "count"], Counter(f for _, _, f in inv["facing_bad"]).most_common(30)))

    an = report["anomalies"]
    out.append("\n## 6. Per-project anomalies\n")
    out.append(f"\n### Replicated stack-chart records ({len(an['replicated_stacks'])})\n")
    out.append("Same unit_type appearing N times with identical specs: the model extracted one unit once per storey row.\n")
    if an["replicated_stacks"]:
        out.append(md_table(["slug", "unit_type", "n_records"],
                            sorted(an["replicated_stacks"], key=lambda x: -x[2])[:30]))
    out.append(f"\n### Phantom plan_types ({len(an['phantom_plan_types'])})\n")
    out.append("plan_type whose unit_type never appears in the project's unit_number_map.\n")
    if an["phantom_plan_types"]:
        c = Counter(an["phantom_plan_types"])
        out.append(md_table(["slug", "unit_type", "count"], [(s, u, c[(s, u)]) for (s, u), _ in c.most_common(30)]))
    out.append(f"\n### Low-confidence projects (< 0.7) ({len(an['low_conf_projects'])})\n")
    if an["low_conf_projects"]:
        out.append(md_table(["slug", "confidence"], sorted(an["low_conf_projects"], key=lambda x: x[1])[:30]))

    cf = report["conflicts"]
    out.append(f"\n## 7. unit_number conflict leaderboard\n\nTotal conflicts: **{cf['n_total']}**.\n")
    if cf["top_projects"]:
        out.append(md_table(["slug", "n_conflicts"], cf["top_projects"]))
    return "\n".join(out)


def run(input_path: Path, output_md: Path) -> dict:
    records = list(iter_jsonl(input_path))
    report = profile(records)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render(report, input_path.name), encoding="utf-8")
    output_md.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
