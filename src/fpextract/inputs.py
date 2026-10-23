"""Input contract: where the floor-plan images come from.

``fpextract`` is deliberately agnostic about how images were obtained.
Two layouts are accepted everywhere a ``--input`` is taken:

1. **Manifest JSONL** — one JSON object per project::

       {"project": "harbour-view", "title": "Harbour View",
        "images": ["plans/hv-a1.png", "plans/hv-b2.png"],
        "url": "https://example.com/harbour-view",        # optional
        "description": "2 Bedroom, 3 Bedroom, Penthouse", # optional, free text
        "property_type": "condo_apartment"}               # optional hint

   Relative image paths resolve against the manifest file's directory.

2. **Directory tree** — ``<root>/<project>/*.{jpg,jpeg,png,webp}``. The
   folder name is the project slug and title.

Both yield :class:`FloorPlanRef` objects and a project-metadata map.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class FloorPlanRef:
    """One image belonging to one project."""
    project: str
    image_path: Path


@dataclass
class ProjectMeta:
    slug: str
    title: str = ""
    url: str = ""
    description: str = ""
    property_type: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"title": self.title or self.slug, "url": self.url,
                "description": self.description, "property_type": self.property_type,
                **self.extra}


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed objects from a JSONL file, skipping blank/malformed lines."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _is_manifest(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".jsonl", ".ndjson", ".json"}


def iter_floor_plans(source: Path, projects_filter: set[str] | None = None) -> Iterator[FloorPlanRef]:
    """Yield ``FloorPlanRef`` for every image in a manifest or directory tree.

    Missing files are skipped silently so a partially downloaded corpus still
    runs; the caller's "done" bookkeeping is keyed by path so re-runs resume.
    """
    source = Path(source)
    if not source.exists():
        raise SystemExit(f"input not found: {source}")
    if _is_manifest(source):
        base = source.parent
        for rec in iter_jsonl(source):
            slug = (rec.get("project") or rec.get("slug") or "").strip()
            if not slug or (projects_filter and slug not in projects_filter):
                continue
            for raw in rec.get("images") or []:
                if not raw:
                    continue
                p = Path(raw)
                if not p.is_absolute():
                    p = (base / p).resolve()
                if p.exists() and p.suffix.lower() in IMAGE_SUFFIXES:
                    yield FloorPlanRef(slug, p)
        return
    if source.is_dir():
        for proj_dir in sorted(p for p in source.iterdir() if p.is_dir()):
            slug = proj_dir.name
            if projects_filter and slug not in projects_filter:
                continue
            for img in sorted(proj_dir.iterdir()):
                if img.is_file() and img.suffix.lower() in IMAGE_SUFFIXES:
                    yield FloorPlanRef(slug, img.resolve())
        return
    raise SystemExit(f"input must be a manifest .jsonl or a directory tree: {source}")


def load_project_meta(source: Path) -> dict[str, ProjectMeta]:
    """Slug → metadata. For a directory tree the slug doubles as the title."""
    source = Path(source)
    out: dict[str, ProjectMeta] = {}
    if not source.exists():
        return out
    if _is_manifest(source):
        for rec in iter_jsonl(source):
            slug = (rec.get("project") or rec.get("slug") or "").strip()
            if not slug:
                continue
            known = {"project", "slug", "images", "title", "url", "description", "property_type"}
            out[slug] = ProjectMeta(
                slug=slug,
                title=rec.get("title") or slug,
                url=rec.get("url") or "",
                description=rec.get("description") or "",
                property_type=rec.get("property_type") or None,
                extra={k: v for k, v in rec.items() if k not in known},
            )
        return out
    if source.is_dir():
        for proj_dir in sorted(p for p in source.iterdir() if p.is_dir()):
            out[proj_dir.name] = ProjectMeta(slug=proj_dir.name, title=proj_dir.name)
    return out


def parse_projects_only(value: str | None) -> set[str] | None:
    """``"a,b, c"`` → ``{"a", "b", "c"}``; empty → ``None`` (no filter)."""
    if not value:
        return None
    s = {x.strip() for x in value.split(",") if x.strip()}
    return s or None


def write_manifest(path: Path, projects: Iterable[tuple[str, list[Path], dict]]) -> None:
    """Helper for tests and converters: write ``(slug, [paths], meta)`` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for slug, paths, meta in projects:
            row = {"project": slug, **meta, "images": [str(p) for p in paths]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
