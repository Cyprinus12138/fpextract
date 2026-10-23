"""Optional Postgres sink (``pip install fpextract[pg]``).

Writes each project atomically: parent upsert plus delete-and-insert of its
unit rows inside one transaction, so the parent can never point at a stale
child set. Schema lives in ``schema.sql`` next to this module; apply it once
with ``fpextract sync-pg --init-schema`` or ``psql``.
"""
from __future__ import annotations

import json
import logging
import os
from importlib import resources
from pathlib import Path

from fpextract.sink.jsonl import iter_jsonl


def connect_from_env():
    try:
        import psycopg
    except ImportError as e:  # pragma: no cover
        raise ImportError("psycopg is required: pip install 'fpextract[pg]'") from e
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("FPEXTRACT_DB_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL (or FPEXTRACT_DB_URL) not set")
    return psycopg.connect(dsn, autocommit=True)


def schema_sql() -> str:
    return resources.files("fpextract.sink").joinpath("schema.sql").read_text(encoding="utf-8")


def init_schema(conn) -> None:
    conn.execute(schema_sql())


def upsert_project(conn, rec: dict) -> int:
    """Upsert one aggregated project record. Returns the number of unit rows."""
    slug = rec.get("project_slug") or ""
    plan_types = rec.get("plan_types") or []
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO floorplan_projects
              (slug, title, url, n_source_images, n_physical_units, n_multi_storey_units,
               dominant_property_type, property_type_source, property_types_seen, confidence,
               unit_number_map, unit_number_evidence, conflicts, aggregated_at, synced_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, now())
            ON CONFLICT (slug) DO UPDATE SET
              title = EXCLUDED.title, url = EXCLUDED.url,
              n_source_images = EXCLUDED.n_source_images,
              n_physical_units = EXCLUDED.n_physical_units,
              n_multi_storey_units = EXCLUDED.n_multi_storey_units,
              dominant_property_type = EXCLUDED.dominant_property_type,
              property_type_source = EXCLUDED.property_type_source,
              property_types_seen = EXCLUDED.property_types_seen,
              confidence = EXCLUDED.confidence,
              unit_number_map = EXCLUDED.unit_number_map,
              unit_number_evidence = EXCLUDED.unit_number_evidence,
              conflicts = EXCLUDED.conflicts,
              aggregated_at = EXCLUDED.aggregated_at, synced_at = now()
            """,
            (slug, rec.get("project_title"), rec.get("project_url"),
             int(rec.get("n_source_images") or 0), rec.get("n_physical_units"),
             rec.get("n_multi_storey_units"), rec.get("dominant_property_type"),
             rec.get("property_type_source"), rec.get("property_types_seen") or None,
             rec.get("confidence"), json.dumps(rec.get("unit_number_map") or {}),
             json.dumps(rec.get("unit_number_evidence") or {}), json.dumps(rec.get("conflicts") or []),
             rec.get("aggregated_at") or None),
        )
        conn.execute("DELETE FROM floorplan_units WHERE slug = %s", (slug,))
        rows = []
        for p in plan_types:
            ut = (p.get("unit_type") or "").strip()
            if not ut:
                continue
            rows.append((slug, ut, p.get("classification"), p.get("property_type"),
                         int(p.get("n_storeys") or 1), p.get("bedrooms"), p.get("bathrooms"),
                         p.get("total_area_sqm"), p.get("balcony_sqm"), p.get("facing"),
                         json.dumps(p.get("rooms") or []), json.dumps(p.get("windows") or []),
                         json.dumps(p["storeys"]) if p.get("storeys") else None,
                         p.get("confidence"), p.get("n_source_records"),
                         p.get("source_image_sha256s") or None, p.get("source_image_basenames") or None,
                         p.get("match_methods") or None))
        if rows:
            conn.cursor().executemany(
                """
                INSERT INTO floorplan_units
                  (slug, unit_type, classification, property_type, n_storeys, bedrooms, bathrooms,
                   total_area_sqm, balcony_sqm, facing, rooms, windows, storeys, confidence,
                   n_source_records, source_image_sha256s, source_image_basenames, match_methods)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                        %s, %s, %s, %s)
                """, rows)
        return len(rows)


def sync(projects_path: Path, log: logging.Logger, *, dry_run: bool = False,
         projects_only: set[str] | None = None, init: bool = False) -> dict:
    if not projects_path.exists():
        raise SystemExit(f"input not found: {projects_path} — run `fpextract aggregate` first")
    conn = None if dry_run else connect_from_env()
    if conn is not None and init:
        init_schema(conn)
        log.info("schema applied")
    totals = {"seen": 0, "upserted": 0, "units": 0, "failed": 0}
    for rec in iter_jsonl(projects_path):
        slug = rec.get("project_slug") or ""
        if not slug or (projects_only and slug not in projects_only):
            continue
        totals["seen"] += 1
        n_units = len(rec.get("plan_types") or [])
        if dry_run:
            log.info(f"  [dry-run] would upsert slug={slug} units={n_units} "
                     f"dominant={rec.get('dominant_property_type')} conf={rec.get('confidence')}")
            continue
        try:
            totals["units"] += upsert_project(conn, rec)
            totals["upserted"] += 1
            log.info(f"[{slug}] units={n_units} conf={rec.get('confidence')}")
        except Exception as e:  # noqa: BLE001
            totals["failed"] += 1
            log.error(f"[{slug}] FAIL: {type(e).__name__}: {e}")
    log.info(f"summary: {totals}")
    return totals
