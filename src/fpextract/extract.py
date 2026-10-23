"""Stage 1 — one structured record per floor-plan image.

Two execution modes with identical output rows:

* **sync** — one API call per image, optionally with a thread pool. Resumable:
  images already present in the output JSONL are skipped.
* **batch** — the provider's batch API (half price, async). The workload is
  chunked to provider limits and every ``batch_id`` is persisted to a state
  file via atomic rename *the moment* ``submit_batch`` returns, so a crash can
  never lose track of submitted work and a re-run never pays for the same
  tokens twice.

Prompting strategy is independent of execution mode: ``tool`` (schema-forced
function call) or ``json`` (plain JSON for VLMs without function calling).
Batch is tool-mode only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from fpextract.images import prepare_image, sha256_file
from fpextract.inputs import FloorPlanRef, iter_floor_plans
from fpextract.llm.client import (
    BatchRequest,
    LLMClient,
    load_extraction_schema,
    load_prompt,
)
from fpextract.llm.coords import COORD_SYSTEMS
from fpextract.sink.jsonl import load_done

MODES = ("tool", "json")

# Per-provider per-batch caps with safety margins. Anthropic's hard limits
# are 100K requests / 256 MB; OpenAI's are 50K / 100 MB. Chunk well under
# both so one oversized image cannot push a chunk over.
PROVIDER_BATCH_LIMITS = {
    "anthropic": {"max_requests": 1000, "max_bytes": 200_000_000},
    "openai": {"max_requests": 500, "max_bytes": 80_000_000},
}
DEFAULT_BATCH_LIMITS = {"max_requests": 500, "max_bytes": 80_000_000}

_COORD_INSTRUCTIONS = {
    "0-1000": ("Output coordinates in the 0-1000 normalized scale, where "
               "[0, 0, 1000, 1000] is the full image from top-left to bottom-right."),
    "qwen3": ("Output coordinates as absolute pixels of the image you were given, "
              "[x_min, y_min, x_max, y_max] from the top-left corner."),
    "fraction": ("Output coordinates as fractions of the image dimensions in [0.0, 1.0], "
                 "where [0, 0, 1, 1] is the full image from top-left to bottom-right."),
}


def json_prompt(coord: str) -> str:
    if coord not in COORD_SYSTEMS:
        raise ValueError(f"coord must be one of {COORD_SYSTEMS}")
    return load_prompt("extract_json").replace("{COORD_INSTRUCTION}", _COORD_INSTRUCTIONS[coord])


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class ExtractConfig:
    input: Path
    output: Path
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    mode: str = "tool"                 # tool | json
    coord: str = "0-1000"              # json mode only
    max_tokens: int = 16384            # json mode only
    no_think: bool = False             # json mode only (vLLM / Qwen3)
    workers: int = 1                   # sync concurrency
    throttle: float = 0.5              # sync, seconds between calls per worker
    limit: int = 0                     # stop after N projects (0 = all)
    projects_only: set[str] | None = None
    reextract: bool = False
    max_side: int = 3000
    # batch
    batch: bool = False
    poll_interval: float = 30.0
    state_path: Path | None = None
    extra: dict = field(default_factory=dict)

    def resolved_state_path(self) -> Path:
        return self.state_path or self.output.parent / ".batch_state.json"


class Extractor:
    """Drives one extraction run. Construct, then call :meth:`run`."""

    def __init__(self, cfg: ExtractConfig, client: LLMClient, log: logging.Logger) -> None:
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if cfg.batch and cfg.mode != "tool":
            raise ValueError("batch runs require --mode tool (provider batch APIs enforce the schema)")
        self.cfg = cfg
        self.client = client
        self.log = log
        self.system_prompt = load_prompt("extract")
        self.tool_schema = load_extraction_schema()
        self.json_user_prompt = json_prompt(cfg.coord)
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------ io
    def _write(self, row: dict) -> None:
        with self._write_lock:
            with self.cfg.output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _error_row(self, slug: str, key: str, error: str, batch_id: str | None = None) -> dict:
        row = {"project_slug": slug, "image_path": key, "error": error,
               "model": self.cfg.model, "mode": self.cfg.mode, "extracted_at": now_iso()}
        if batch_id:
            row["batch_id"] = batch_id
        return row

    def _ok_row(self, slug: str, key: str, record: dict, usage: dict | None,
                batch_id: str | None = None) -> dict:
        row = {"project_slug": slug, "image_path": key, "image_sha256": sha256_file(key),
               **record, "model": self.cfg.model, "mode": self.cfg.mode,
               "extracted_at": now_iso(), "raw_usage": usage or {}}
        if batch_id:
            row["batch_id"] = batch_id
        return row

    def _log_ok(self, slug: str, name: str, record: dict, usage: dict | None, elapsed: float | None) -> None:
        usage = usage or {}
        self.log.info(
            f"[{slug}] {name} OK class={record.get('classification', '?')} "
            f"units={len(record.get('units') or [])} "
            f"unit_numbers={len(record.get('unit_number_map') or {})} "
            f"tokens={usage.get('input_tokens')}/{usage.get('output_tokens')}"
            + (f" [{elapsed:.1f}s]" if elapsed is not None else ""))

    # -------------------------------------------------------------- planning
    def plan(self, done: set[str], exclude: set[str] | None = None) -> tuple[list[FloorPlanRef], int, int]:
        """Return ``(work, n_skipped, n_projects)`` honouring ``limit`` at
        project granularity."""
        seen: set[str] = set()
        work: list[FloorPlanRef] = []
        skipped = 0
        for ref in iter_floor_plans(self.cfg.input, self.cfg.projects_only):
            if self.cfg.limit and ref.project not in seen and len(seen) >= self.cfg.limit:
                break
            seen.add(ref.project)
            key = str(ref.image_path)
            if key in done or (exclude and key in exclude):
                skipped += 1
                continue
            work.append(ref)
        return work, skipped, len(seen)

    def load_done(self) -> set[str]:
        if self.cfg.reextract:
            return set()
        return load_done(self.cfg.output, include_errors=(self.cfg.mode == "tool"))

    # ------------------------------------------------------------ one image
    def extract_one(self, ref: FloorPlanRef) -> tuple[dict, dict, float]:
        t0 = time.monotonic()
        img_bytes, media_type = prepare_image(ref.image_path, max_side=self.cfg.max_side)
        if self.cfg.mode == "tool":
            record, usage = self.client.extract_tool(img_bytes, media_type, self.cfg.model,
                                                     self.system_prompt, self.tool_schema)
        else:
            record, usage = self.client.extract_json(img_bytes, media_type, self.cfg.model,
                                                     self.json_user_prompt, coord=self.cfg.coord,
                                                     max_tokens=self.cfg.max_tokens,
                                                     no_think=self.cfg.no_think)
        return record, usage, time.monotonic() - t0

    # ------------------------------------------------------------------ sync
    def run_sync(self) -> dict:
        done = self.load_done()
        work, n_skipped, n_projects = self.plan(done)
        self.log.info(f"work queue: {len(work)} images, {n_skipped} already done, "
                      f"{n_projects} projects, workers={self.cfg.workers}")
        n_ok = n_fail = 0

        def _job(ref: FloorPlanRef):
            result = self.extract_one(ref)
            if self.cfg.throttle:
                time.sleep(self.cfg.throttle)
            return result

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.workers)) as pool:
            futures = {pool.submit(_job, ref): ref for ref in work}
            for fut in as_completed(futures):
                ref = futures[fut]
                key = str(ref.image_path)
                try:
                    record, usage, elapsed = fut.result()
                except Exception as e:  # noqa: BLE001 — every failure becomes a row
                    self.log.error(f"[{ref.project}] {ref.image_path.name} FAIL ({type(e).__name__}: {e})")
                    self._write(self._error_row(ref.project, key, f"{type(e).__name__}: {e}"))
                    n_fail += 1
                    continue
                self._write(self._ok_row(ref.project, key, record, usage))
                self._log_ok(ref.project, ref.image_path.name, record, usage, elapsed)
                n_ok += 1
        summary = {"extracted": n_ok, "failed": n_fail, "skipped": n_skipped, "projects": n_projects}
        self.log.info(f"summary[sync]: {summary} -> {self.cfg.output}")
        return summary

    # ----------------------------------------------------------------- batch
    @staticmethod
    def custom_id_for(image_path: Path | str) -> str:
        """Deterministic id that fits provider ``custom_id`` length limits."""
        return hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def state_write(path: Path, state: dict) -> None:
        """Temp file + ``os.replace`` so a crash mid-write never leaves a
        half-written state file. Runs right after every successful submit."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def state_load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _stream_chunks(self, work: list[FloorPlanRef]):
        """Yield ``(requests, custom_id_to_path)`` chunks sized to provider
        limits, preparing images lazily so only one chunk is in memory."""
        limits = PROVIDER_BATCH_LIMITS.get(self.cfg.provider, DEFAULT_BATCH_LIMITS)
        reqs: list[BatchRequest] = []
        paths: dict[str, list] = {}
        nbytes = 0
        for ref in work:
            try:
                img_bytes, media_type = prepare_image(ref.image_path, max_side=self.cfg.max_side)
            except Exception as e:  # noqa: BLE001
                self.log.error(f"[{ref.project}] {ref.image_path.name} prepare FAIL: {type(e).__name__}: {e}")
                self._write(self._error_row(ref.project, str(ref.image_path),
                                            f"prepare_image: {type(e).__name__}: {e}"))
                continue
            if reqs and (len(reqs) >= limits["max_requests"] or nbytes + len(img_bytes) > limits["max_bytes"]):
                yield reqs, paths
                reqs, paths, nbytes = [], {}, 0
            cid = self.custom_id_for(ref.image_path)
            paths[cid] = [ref.project, str(ref.image_path)]
            reqs.append(BatchRequest(custom_id=cid, image_bytes=img_bytes, media_type=media_type,
                                     model=self.cfg.model, system_prompt=self.system_prompt,
                                     tool_schema=self.tool_schema))
            nbytes += len(img_bytes)
        if reqs:
            yield reqs, paths

    def _submit_pending(self, state: dict, done: set[str]) -> int:
        in_flight = {entry[1] for b in state["in_flight"] for entry in b["custom_id_to_path"].values()}
        work, _, n_projects = self.plan(done, exclude=in_flight)
        if not work:
            self.log.info(f"no new images to submit (in-flight={len(state['in_flight'])} done={len(done)})")
            return 0
        self.log.info(f"plan: {len(work)} new images across {n_projects} projects")
        state_path = self.cfg.resolved_state_path()
        n = 0
        for reqs, paths in self._stream_chunks(work):
            size_mb = sum(len(r.image_bytes) for r in reqs) / 1_000_000
            self.log.info(f"submitting chunk: {len(reqs)} reqs / {size_mb:.1f} MB")
            # SAFETY-CRITICAL window: submit and the state write happen
            # back-to-back with no other I/O in between.
            batch_id = self.client.submit_batch(reqs)
            state["in_flight"].append({
                "batch_id": batch_id, "submitted_at": now_iso(), "n_requests": len(reqs),
                "size_mb": round(size_mb, 1), "custom_id_to_path": paths,
            })
            self.state_write(state_path, state)
            self.log.info(f"  -> submitted {batch_id}; state persisted ({len(state['in_flight'])} in flight)")
            n += 1
        return n

    def _collect_one(self, batch: dict) -> int:
        batch_id, mapping = batch["batch_id"], batch["custom_id_to_path"]
        n_ok = n_fail = 0
        for result in self.client.fetch_batch_results(batch_id):
            m = mapping.get(result.custom_id)
            if m is None:
                self.log.warning(f"batch {batch_id}: unknown custom_id={result.custom_id}")
                continue
            slug, key = m
            if not result.succeeded:
                self._write(self._error_row(slug, key, result.error or "unknown error", batch_id))
                self.log.error(f"[{slug}] {Path(key).name} FAIL: {result.error}")
                n_fail += 1
                continue
            self._write(self._ok_row(slug, key, result.record or {}, result.usage, batch_id))
            self._log_ok(slug, Path(key).name, result.record or {}, result.usage, None)
            n_ok += 1
        self.log.info(f"  collected batch {batch_id}: ok={n_ok} fail={n_fail}")
        return n_ok + n_fail

    def _poll_and_collect(self, state: dict) -> None:
        state_path = self.cfg.resolved_state_path()
        while state["in_flight"]:
            finished: list[int] = []
            for i, batch in enumerate(state["in_flight"]):
                try:
                    status = self.client.poll_batch(batch["batch_id"])
                except Exception as e:  # noqa: BLE001 — transient; retry next cycle
                    self.log.warning(f"poll error for {batch['batch_id']}: {type(e).__name__}: {e}")
                    continue
                self.log.info(f"batch {batch['batch_id']} ({batch['n_requests']} reqs): "
                              f"{status.raw_status} {status.counts}")
                if status.status == "completed":
                    try:
                        self._collect_one(batch)
                        finished.append(i)
                    except Exception as e:  # noqa: BLE001
                        self.log.error(f"collect error for {batch['batch_id']}: {type(e).__name__}: {e}")
                elif status.status == "failed":
                    self.log.error(f"batch {batch['batch_id']} terminal failure ({status.raw_status}); "
                                   "dropping from state — affected images stay un-extracted")
                    finished.append(i)
            if finished:
                for i in sorted(finished, reverse=True):
                    state["in_flight"].pop(i)
                self.state_write(state_path, state)
            if state["in_flight"]:
                time.sleep(self.cfg.poll_interval)
        if state_path.exists():
            state_path.unlink()
            self.log.info(f"all batches collected; removed state file {state_path}")

    def run_batch(self) -> dict:
        state_path = self.cfg.resolved_state_path()
        if state_path.exists():
            state = self.state_load(state_path)
            if state.get("provider") != self.cfg.provider:
                raise SystemExit(f"state file {state_path} is for provider={state.get('provider')!r}; "
                                 "finish or delete it before switching provider")
            self.log.info(f"loaded state: {len(state['in_flight'])} batches in flight "
                          f"(started {state.get('started_at')})")
        else:
            state = {"provider": self.cfg.provider, "model": self.cfg.model,
                     "started_at": now_iso(), "in_flight": []}
        done = self.load_done()
        n_new = self._submit_pending(state, done)
        self.log.info(f"submission phase complete: {n_new} new batch(es), "
                      f"{len(state['in_flight'])} in flight total")
        if not state["in_flight"]:
            if state_path.exists():
                state_path.unlink()
            return {"submitted": n_new, "collected": 0}
        self._poll_and_collect(state)
        return {"submitted": n_new, "collected": 1}

    def run(self) -> dict:
        self.cfg.output.parent.mkdir(parents=True, exist_ok=True)
        return self.run_batch() if self.cfg.batch else self.run_sync()
