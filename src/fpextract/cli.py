"""``fpextract`` command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fpextract import __version__
from fpextract.inputs import parse_projects_only
from fpextract.llm.client import PROVIDERS, api_key_from_env, make_client, resolve_model
from fpextract.llm.coords import COORD_SYSTEMS
from fpextract.log import log_run_finished, setup_logger


def _add_provider_args(p: argparse.ArgumentParser, default_model: str = "sonnet",
                       default_provider: str = "anthropic") -> None:
    p.add_argument("--provider", default=default_provider, choices=PROVIDERS,
                   help=f"LLM provider (default: {default_provider})")
    p.add_argument("--model", default=default_model,
                   help=f"Model id or alias sonnet|opus|haiku (default: {default_model})")
    p.add_argument("--base-url", default=None,
                   help="Base URL for OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, DashScope, Azure). "
                        "Requires --provider openai.")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout in seconds (default 120; use 30-60 for local endpoints)")


def _client(args):
    if args.base_url and args.provider != "openai":
        raise SystemExit("--base-url requires --provider openai")
    api_key = api_key_from_env(args.provider, args.base_url)
    client = make_client(args.provider, api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    return client, resolve_model(args.provider, args.model)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fpextract",
                                 description="Structured floor-plan extraction with vision LLMs.")
    ap.add_argument("--version", action="version", version=f"fpextract {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # extract ---------------------------------------------------------------
    p = sub.add_parser("extract", help="Stage 1: one structured record per image (sync or batch)")
    p.add_argument("--input", type=Path, required=True,
                   help="Manifest .jsonl or a directory tree <root>/<project>/*.png (see samples/README.md)")
    p.add_argument("--output", type=Path, default=Path("output/extractions.jsonl"))
    p.add_argument("--log", type=Path, default=None, help="Append-mode log file (default: stdout only)")
    _add_provider_args(p)
    p.add_argument("--mode", default="tool", choices=["tool", "json"],
                   help="tool = schema-forced function call (default; required for --batch); "
                        "json = plain-JSON prompt for VLMs without function calling (Qwen-VL, LLaVA, …)")
    p.add_argument("--coord", default="0-1000", choices=COORD_SYSTEMS,
                   help="json mode: coordinate system the model emits (0-1000 for Qwen2.5/DashScope, "
                        "qwen3 for local Qwen3-VL on vLLM, fraction for Claude-style models)")
    p.add_argument("--max-tokens", type=int, default=16384, help="json mode: output token budget")
    p.add_argument("--no-think", action="store_true",
                   help="json mode: disable Qwen3 chain-of-thought on vLLM (enable_thinking=false)")
    p.add_argument("--workers", type=int, default=1, help="sync mode: concurrent API threads (default 1)")
    p.add_argument("--throttle", type=float, default=0.5, help="sync mode: seconds between calls per worker")
    p.add_argument("--limit", type=int, default=0, help="Stop after N projects (0 = all)")
    p.add_argument("--projects-only", default="", help="Comma-separated project slugs to process")
    p.add_argument("--reextract", action="store_true", help="Ignore the done set and re-extract every match")
    p.add_argument("--max-side", type=int, default=3000, help="Long-edge pixel cap for the sent image")
    p.add_argument("--batch", action="store_true",
                   help="Use the provider batch API (half price, async). Auto-chunks to provider limits and "
                        "persists every batch_id so an interrupted run resumes without re-spending tokens.")
    p.add_argument("--poll-interval", type=float, default=30.0, help="batch: seconds between status polls")
    p.add_argument("--state", type=Path, default=None,
                   help="batch: state file (default: <output dir>/.batch_state.json)")
    p.set_defaults(func=cmd_extract)

    # aggregate -------------------------------------------------------------
    p = sub.add_parser("aggregate", help="Stage 2: roll per-image rows up to per-project records (no API spend)")
    p.add_argument("--extractions", type=Path, default=Path("output/extractions.jsonl"))
    p.add_argument("--input", type=Path, default=None,
                   help="The manifest / directory used for extract, for titles, urls and property hints")
    p.add_argument("--sheet-titles", type=Path, default=None,
                   help="Optional per-image sidecar JSONL with classification / plan_types / storey_label "
                        "(see docs/PIPELINE.md)")
    p.add_argument("--output", type=Path, default=Path("output/projects.jsonl"))
    p.add_argument("--projects-only", default="")
    p.add_argument("--log", type=Path, default=None)
    p.set_defaults(func=cmd_aggregate)

    # classify --------------------------------------------------------------
    p = sub.add_parser("classify", help="Sidecar audit: sheet type and where the unit code lives (image vs filename)")
    p.add_argument("--extractions", type=Path, default=Path("output/extractions.jsonl"))
    p.add_argument("--output", type=Path, default=Path("output/classifications.jsonl"))
    _add_provider_args(p)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Stop after N projects (0 = all)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--projects-only", default="")
    p.add_argument("--reclassify", action="store_true")
    p.add_argument("--log", type=Path, default=None)
    p.set_defaults(func=cmd_classify)

    # try-one / try-crop / try-norm -----------------------------------------
    p = sub.add_parser("try-one", help="One image → extraction HTML in your browser (nothing written to output/)")
    p.add_argument("image", type=Path)
    _add_provider_args(p)
    p.add_argument("--crop-pad", type=float, default=0.02, help="Fractional margin added to each crop edge")
    p.add_argument("--max-side", type=int, default=3000)
    p.add_argument("--out", type=Path, default=None, help="Write HTML here instead of a temp file")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_try_one)

    p = sub.add_parser("try-crop", help="One image → LLM crop vs OpenCV-refined crop HTML (needs fpextract[cv])")
    p.add_argument("image", type=Path)
    _add_provider_args(p)
    p.add_argument("--crop-pad", type=float, default=0.02)
    p.add_argument("--snap-margin", type=float, default=0.015, help="Margin kept around detected content")
    p.add_argument("--max-side", type=int, default=3000)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_try_crop)

    p = sub.add_parser("try-norm", help="Background normalisation / watermark removal HTML (no API cost)")
    p.add_argument("image", type=Path)
    p.add_argument("--crop", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Crop fractions in [0,1], e.g. a crop_frac from an extraction")
    p.add_argument("--crop-pad", type=float, default=0.02)
    p.add_argument("--bg-kernel-frac", type=float, default=0.07)
    p.add_argument("--tiled-watermark", action="store_true", help="Also run tiled text-watermark removal")
    p.add_argument("--slbr", type=Path, default=None, metavar="WEIGHTS", help="SLBR weights (.pth); see docs")
    p.add_argument("--out-image", type=Path, default=None, help="Save the cleaned image here")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_try_norm)

    # dq --------------------------------------------------------------------
    p = sub.add_parser("dq-profile", help="Read-only data-quality report for projects.jsonl")
    p.add_argument("--input", type=Path, default=Path("output/projects.jsonl"))
    p.add_argument("--output", type=Path, default=Path("output/dq/report.md"))
    p.set_defaults(func=cmd_dq_profile)

    p = sub.add_parser("dq-clean", help="Apply idempotent cleaning rules to projects.jsonl")
    p.add_argument("--input", type=Path, default=Path("output/projects.jsonl"))
    p.add_argument("--output", type=Path, default=Path("output/projects.cleaned.jsonl"))
    p.add_argument("--report", type=Path, default=Path("output/dq/cleaning-report.md"))
    p.set_defaults(func=cmd_dq_clean)

    # sync-pg ---------------------------------------------------------------
    p = sub.add_parser("sync-pg", help="Upsert projects.jsonl into Postgres (needs fpextract[pg], DATABASE_URL)")
    p.add_argument("--input", type=Path, default=Path("output/projects.jsonl"))
    p.add_argument("--init-schema", action="store_true", help="Apply sink/schema.sql first")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--projects-only", default="")
    p.add_argument("--log", type=Path, default=None)
    p.set_defaults(func=cmd_sync_pg)
    return ap


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_extract(args) -> int:
    from fpextract.extract import ExtractConfig, Extractor

    if args.batch and args.base_url:
        raise SystemExit("--base-url is not supported with --batch (provider batch APIs only)")
    if args.batch and args.mode != "tool":
        raise SystemExit("--batch requires --mode tool")
    log = setup_logger("extract", args.log)
    client, model = _client(args)
    cfg = ExtractConfig(
        input=args.input, output=args.output, provider=args.provider, model=model, mode=args.mode,
        coord=args.coord, max_tokens=args.max_tokens, no_think=args.no_think, workers=args.workers,
        throttle=args.throttle, limit=args.limit, projects_only=parse_projects_only(args.projects_only),
        reextract=args.reextract, max_side=args.max_side, batch=args.batch,
        poll_interval=args.poll_interval, state_path=args.state,
    )
    log.info(f"config: provider={args.provider} model={model} mode={args.mode} "
             f"{'batch' if args.batch else 'sync'} input={args.input} output={args.output}")
    summary = Extractor(cfg, client, log).run()
    log_run_finished(log)
    print(json.dumps(summary))
    return 0


def cmd_aggregate(args) -> int:
    from fpextract.aggregate import aggregate
    from fpextract.sink.jsonl import write_jsonl

    log = setup_logger("aggregate", args.log)
    records, stats = aggregate(args.extractions, input_source=args.input, sidecar=args.sheet_titles,
                               projects_only=parse_projects_only(args.projects_only))
    n = write_jsonl(args.output, records)
    log.info(f"wrote {n} project records -> {args.output}; {stats}")
    log_run_finished(log)
    return 0


def cmd_classify(args) -> int:
    from fpextract import classify

    log = setup_logger("classify", args.log)
    client, model = _client(args)
    classify.run(args.extractions, args.output, client, model, log, workers=args.workers, limit=args.limit,
                 projects_only=parse_projects_only(args.projects_only), reclassify=args.reclassify,
                 max_tokens=args.max_tokens, no_think=args.no_think)
    log_run_finished(log)
    return 0


def _print_client(args, client, model) -> None:
    print(f"image    : {args.image.resolve()}")
    print(f"provider : {args.provider}")
    print(f"base-url : {client.effective_base_url}" + (" (from --base-url)" if args.base_url else " (SDK default / env)"))
    print(f"model    : {model}" + (f"  (resolved from alias '{args.model}')" if args.model != model else ""))


def cmd_try_one(args) -> int:
    from fpextract.verify import try_one

    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")
    client, model = _client(args)
    _print_client(args, client, model)
    path, record = try_one(args.image, client, model, crop_pad=args.crop_pad, max_side=args.max_side,
                           out=args.out, open_browser=not args.no_browser)
    units = record.get("units") or []
    print(f"done     : class={record.get('classification')!r} conf={record.get('confidence')!r} "
          f"units={len(units)} crops={sum(1 for u in units if u.get('crop_frac'))}")
    print(f"HTML     : {path}")
    return 0


def cmd_try_crop(args) -> int:
    from fpextract.verify import try_crop

    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")
    client, model = _client(args)
    _print_client(args, client, model)
    path, record, refined = try_crop(args.image, client, model, crop_pad=args.crop_pad,
                                     snap_margin=args.snap_margin, max_side=args.max_side,
                                     out=args.out, open_browser=not args.no_browser)
    for i, (u, rf) in enumerate(zip(record.get("units") or [], refined, strict=False)):
        frac = u.get("crop_frac")
        if frac and rf:
            print(f"  unit {i}: LLM [{' '.join(f'{v:.3f}' for v in frac)}] -> CV [{' '.join(f'{v:.3f}' for v in rf)}]")
    print(f"HTML     : {path}")
    return 0


def cmd_try_norm(args) -> int:
    from fpextract.verify import try_norm

    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")
    path = try_norm(args.image, crop=args.crop, crop_pad=args.crop_pad, bg_kernel_frac=args.bg_kernel_frac,
                    tiled_watermark=args.tiled_watermark, slbr_weights=args.slbr, out_image=args.out_image,
                    out=args.out, open_browser=not args.no_browser)
    print(f"HTML     : {path}")
    if args.out_image:
        print(f"saved    : {args.out_image}")
    return 0


def cmd_dq_profile(args) -> int:
    from fpextract.dq import profile

    report = profile.run(args.input, args.output)
    print(f"wrote {args.output} and {args.output.with_suffix('.json')}")
    print(json.dumps(report["totals"]))
    return 0


def cmd_dq_clean(args) -> int:
    from fpextract.dq import clean

    summary = clean.run(args.input, args.output, args.report)
    print(json.dumps(summary))
    print(f"wrote {args.output} and {args.report}")
    return 0


def cmd_sync_pg(args) -> int:
    from fpextract.sink import postgres

    log = setup_logger("sync-pg", args.log)
    postgres.sync(args.input, log, dry_run=args.dry_run, projects_only=parse_projects_only(args.projects_only),
                  init=args.init_schema)
    log_run_finished(log)
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))
