"""Append-only JSONL conventions used by every stage.

* An image row carries ``image_path`` plus either the extraction fields or
  an ``error`` key. The done-set is keyed by ``image_path`` so re-runs skip
  work already recorded.
* Project rows carry ``project_slug``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_done(path: Path, projects_filter: set[str] | None = None,
              include_errors: bool = True) -> set[str]:
    """Image paths already present in ``path``. With ``include_errors=False``
    only successful rows count, so failed images are retried on re-run."""
    done: set[str] = set()
    for rec in iter_jsonl(path):
        if "image_path" not in rec:
            continue
        if not include_errors and "error" in rec:
            continue
        if projects_filter is not None and rec.get("project_slug") not in projects_filter:
            continue
        done.add(rec["image_path"])
    return done


def load_extractions(path: Path) -> dict[str, list[dict]]:
    """Group successful per-image records by ``project_slug``.

    Skips error rows. When the same ``image_path`` appears more than once
    (resume + re-extract) the most recent ``extracted_at`` wins.
    """
    if not path.exists():
        raise SystemExit(f"extractions not found: {path} — run `fpextract extract` first")
    by_image: dict[str, dict] = {}
    for rec in iter_jsonl(path):
        if rec.get("error"):
            continue
        ip = rec.get("image_path")
        if not ip:
            continue
        prior = by_image.get(ip)
        if prior is None or rec.get("extracted_at", "") >= prior.get("extracted_at", ""):
            by_image[ip] = rec
    by_project: dict[str, list[dict]] = defaultdict(list)
    for rec in by_image.values():
        by_project[rec.get("project_slug") or "unknown"].append(rec)
    return by_project
