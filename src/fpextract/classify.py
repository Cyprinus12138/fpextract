"""Sidecar pass: audit each image's *type* and whether its unit-type code is
printed on the sheet or only encoded in the filename.

The extractor sees the image only. For projects where the unit code lives
in the filename (``1-Bedroom-a4-b1.jpg``) the vision model either returns a
generic descriptor ("1-Bedroom") or repeats one label across distinct
images. This pass sees both the image and the filename stem and can say
which is authoritative. Output is content-addressed by ``image_sha256`` so
re-runs are no-ops.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fpextract.images import prepare_image
from fpextract.llm.client import LLMClient, parse_json_reply
from fpextract.sink.jsonl import iter_jsonl

VALID_CLASSES = {
    "per_unit_plan",      # one unit's plan with room-level detail
    "stack_chart",        # storey × stack table mapping unit numbers → types
    "site_plan",          # bird's-eye site map
    "floor_plate",        # one entire storey with all units, codes only
    "diagrammatic_chart", # coloured per-floor block diagram
    "brochure_cover",     # cover page / 3D render / amenity photo
    "other",
}
NON_UNIT_CLASSES = {"site_plan", "stack_chart", "diagrammatic_chart", "brochure_cover", "other"}

SYSTEM = ("You are a precise floor-plan auditor. "
          "Return valid JSON only — no prose, no markdown fences.")

PROMPT_TEMPLATE = """\
You are auditing one image from a residential project's marketing materials.

The image filename (basename, no extension) is: {stem}

Return a single JSON object with this exact structure — no extra keys, no markdown:

{{
  "classification": "<per_unit_plan|stack_chart|site_plan|floor_plate|diagrammatic_chart|brochure_cover|other>",
  "has_plan_type_in_picture":  <true|false>,
  "has_plan_type_in_filename": <true|false>,
  "plan_type": "<short identifier code>" | null
}}

Definitions
-----------
- per_unit_plan       : a single unit's floor plan with detailed rooms; usually a unit_type
                        label like "TYPE A1" or "B2-G" is printed in the title block or corner.
- stack_chart         : a column-and-row table mapping unit numbers (e.g. #02-22) to unit
                        types across building storeys. NOT a per-unit drawing.
- site_plan           : a top-down map of the whole development showing block locations.
- floor_plate         : a diagram of one entire storey of a building showing all units on
                        that floor labelled by code (A1, B1, …) but limited room detail per unit.
- diagrammatic_chart  : a coloured stack diagram showing unit positions per floor.
- brochure_cover      : a marketing cover page, 3D render, or amenity photo with no plan content.
- other               : anything else.

Field rules
-----------
- has_plan_type_in_picture  : TRUE only when a unit_type code is VISIBLY PRINTED inside
                              the image (title block, corner label, etc.). Bedroom counts
                              alone ("1-BEDROOM", "2BR") do NOT count — those are descriptors,
                              not unit_type codes. The code must look like an alphanumeric
                              identifier (e.g. "A1", "B2", "a4b1", "C3a", "TYPE B").
- has_plan_type_in_filename : TRUE only when the filename contains a clear alphanumeric
                              unit_type code. Pure bedroom descriptors ("1-Bedroom"),
                              sequence numbers ("3", "12"), and positional tags
                              ("Lower-Level", "Highrise") do NOT count.
- plan_type                 : the canonical unit_type. Prefer the value from the picture;
                              fall back to the filename; otherwise null. Preserve the
                              casing as printed/encoded.
- classification not in {{per_unit_plan, floor_plate}} ⇒ plan_type MUST be null.

Examples
--------
- filename "1-Bedroom-a4-b1-scaled", image shows one unit's plan, no on-image type label
  → classification=per_unit_plan, picture=false, filename=true, plan_type="a4b1"
- filename "Type-B2-Highrise", image clearly shows "TYPE B2" in the corner
  → picture=true, filename=true, plan_type="B2"
- filename "Diagrammatic-Chart-scaled", image is a coloured per-floor diagram
  → classification=diagrammatic_chart, both false, plan_type=null
- filename "2-bedroom-3-scaled", image shows one unit with "TYPE C1" printed
  → picture=true, filename=false (just a sequence number), plan_type="C1"
"""


def validate_classification(rec: dict) -> dict:
    """Coerce the model output to the contract; raise on irrecoverable errors."""
    cls = rec.get("classification")
    if cls not in VALID_CLASSES:
        raise ValueError(f"invalid classification: {cls!r}")
    has_pic = bool(rec.get("has_plan_type_in_picture"))
    has_file = bool(rec.get("has_plan_type_in_filename"))
    plan = rec.get("plan_type")
    if plan is not None and not isinstance(plan, str):
        raise ValueError(f"plan_type must be string|null, got {type(plan).__name__}")
    if plan is not None:
        plan = plan.strip() or None
    if cls in NON_UNIT_CLASSES:
        plan, has_pic, has_file = None, False, False
    return {"classification": cls, "has_plan_type_in_picture": has_pic,
            "has_plan_type_in_filename": has_file, "plan_type": plan}


def load_done(out_path: Path) -> set[str]:
    return {r["image_sha256"] for r in iter_jsonl(out_path)
            if r.get("image_sha256") and "error" not in r}


def iter_inputs(extractions: Path, projects_filter: set[str] | None,
                done: set[str]) -> list[tuple[str, Path, str]]:
    """Unique ``(slug, image_path, sha)`` from an extractions JSONL, minus done."""
    if not extractions.exists():
        raise SystemExit(f"input not found: {extractions}")
    seen: set[str] = set()
    out: list[tuple[str, Path, str]] = []
    for rec in iter_jsonl(extractions):
        sha, ip = rec.get("image_sha256"), rec.get("image_path")
        slug = rec.get("project_slug") or "unknown"
        if not sha or not ip or (projects_filter is not None and slug not in projects_filter):
            continue
        if sha in seen or sha in done:
            continue
        seen.add(sha)
        out.append((slug, Path(ip), sha))
    return out


def classify_image(client: LLMClient, model: str, image_path: Path, *, max_tokens: int = 512,
                   no_think: bool = False, max_side: int = 3000) -> tuple[dict, dict]:
    img_bytes, media_type = prepare_image(image_path, max_side=max_side)
    prompt = PROMPT_TEMPLATE.format(stem=image_path.stem)
    raw, usage = client.complete_json_raw(img_bytes, media_type, model, SYSTEM, prompt,
                                          max_tokens, no_think=no_think)
    return validate_classification(parse_json_reply(raw)), usage


def run(extractions: Path, output: Path, client: LLMClient, model: str, log: logging.Logger, *,
        workers: int = 4, limit: int = 0, projects_only: set[str] | None = None,
        reclassify: bool = False, max_tokens: int = 512, no_think: bool = False) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set() if reclassify else load_done(output)
    work = iter_inputs(extractions, projects_only, done)
    if limit:
        seen: set[str] = set()
        capped = []
        for item in work:
            if item[0] not in seen and len(seen) >= limit:
                continue
            seen.add(item[0])
            capped.append(item)
        work = capped
    log.info(f"config: model={model} workers={workers} done={len(done)} todo={len(work)}")

    lock = threading.Lock()
    n_ok = n_fail = 0

    def _write(row: dict) -> None:
        with lock, output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _job(slug, image_path, sha):
        t0 = time.monotonic()
        record, usage = classify_image(client, model, image_path, max_tokens=max_tokens, no_think=no_think)
        return record, usage, time.monotonic() - t0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_job, *item): item for item in work}
        for fut in as_completed(futures):
            slug, image_path, sha = futures[fut]
            base = {"image_sha256": sha, "image_path": str(image_path), "project_slug": slug,
                    "model": model, "classified_at": dt.datetime.now(dt.timezone.utc).isoformat()}
            try:
                record, usage, elapsed = fut.result()
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                log.error(f"[{slug}] {image_path.name} FAIL ({type(e).__name__}: {e})")
                _write({**base, "error": f"{type(e).__name__}: {e}"})
                continue
            n_ok += 1
            _write({**base, **record, "usage": usage})
            if (n_ok + n_fail) % 50 == 0:
                log.info(f"progress: ok={n_ok} fail={n_fail} ({elapsed:.1f}s last)")
    log.info(f"done: ok={n_ok} fail={n_fail}")
    return {"ok": n_ok, "failed": n_fail}


# Filename-derived unit code (shared with dq.clean R7) ---------------------

_SUFFIX_TOKENS = {"scaled", "final", "v1", "v2", "v3", "fp", "floorplan", "floor", "plan", "rev", "revised"}
_DESCRIPTOR_RE = re.compile(
    r"^(\d+\s*-?\s*bedroom|\d+\s*-?\s*br|studio|loft|penthouse|ph|duplex)\b[-_\s]*", re.I)
_REJECT_CODES = {"a", "b", "c", "d", "e", "f"}


def derive_unit_code_from_filename(stem: str) -> str | None:
    """``"1-Bedroom-a4-b1-scaled"`` → ``"a4b1"``; ``None`` when the filename
    does not follow a ``<descriptor>-<code>`` convention. Real unit codes
    mix letters and digits; pure words or pure numbers are rejected."""
    s = stem.strip()
    if not s:
        return None
    parts = re.split(r"[-_\s]+", s)
    while parts and parts[-1].lower() in _SUFFIX_TOKENS:
        parts.pop()
    s = "-".join(parts)
    m = _DESCRIPTOR_RE.match(s)
    if not m:
        return None
    tail = s[m.end():].strip("-_ ").strip()
    if not tail or not re.fullmatch(r"[a-zA-Z0-9-]+", tail):
        return None
    code = tail.replace("-", "").lower()
    if not code or len(code) > 24 or code in _REJECT_CODES:
        return None
    if not (any(c.isalpha() for c in code) and any(c.isdigit() for c in code)):
        return None
    return code
