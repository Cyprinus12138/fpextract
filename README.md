# fpextract

**Structured floor-plan extraction with vision LLMs:** a provider-agnostic
client, batch API with auto-chunking and crash-safe resume, LLM
localisation refined by OpenCV, and project-level rollups with provenance
and confidence.

Extracted from [ProProp](https://proprop.algonavis.com/), an AI property
advisor for Singapore running in production, where this pipeline extracted
structured unit data from about 30,000 floor plans for the advisory
knowledge base.

Give it a folder of floor-plan images; get back one JSON record per image
(classification, units with rooms / areas / bedrooms / facing and a crop
box, unit-number map) and one record per project (plan types with
provenance and confidence, unit-number lookup with evidence and conflicts,
multi-storey units, property type). Everything after the LLM call is local
and regenerable.

## Features

- **Provider-agnostic.** Anthropic, OpenAI, and any OpenAI-compatible
  endpoint (vLLM, Ollama, LM Studio, DashScope, Azure). Two prompting
  strategies: schema-forced **tool** calls, or a plain-**json** prompt for
  VLMs without function calling (Qwen3-VL, Qwen2.5-VL, LLaVA, …) with
  chain-of-thought stripping and per-model bounding-box normalisation.
- **Batch API with crash-safe resume.** Half price, auto-chunked to provider
  limits, every `batch_id` persisted by atomic rename the moment it is
  returned. Interrupt it at any point and run the same command again; no
  batch is ever re-submitted.
- **LLM localisation refined by OpenCV.** The model gives a rough crop per
  unit; OpenCV snaps it to the drawing's actual content on white, dark or
  branded backgrounds. Plus background normalisation and weight-free tiled
  watermark removal.
- **Rollups you can audit.** Multi-storey units summed with a per-level
  breakdown, consensus merging across duplicate sheets, unit-number votes
  with `unit_number_evidence` and `conflicts`, confidence at unit and
  project level, and a data-quality profiler + idempotent cleaner.
- **Not tied to how images were obtained.** Input is a directory tree or a manifest JSONL.
  Output is JSONL, optionally Postgres.

## Quickstart

```bash
pip install fpextract            # core: anthropic + pillow
pip install 'fpextract[cv]'      # + OpenCV for try-crop / try-norm
pip install 'fpextract[openai]'  # + OpenAI-compatible endpoints
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# 1. One image → HTML in your browser (crop boxes, rooms, raw JSON). Nothing is written to output/.
fpextract try-one samples/harbour-view/harbour-view-type-a1.png

# 2. Extract a folder (<root>/<project>/*.png). Append-safe: re-runs skip finished images.
fpextract extract --input samples --output out/extractions.jsonl

# 3. Roll up per project. Local, free, regenerable.
fpextract aggregate --extractions out/extractions.jsonl --input samples --output out/projects.jsonl

# 4. Full corpus at half price, resumable.
fpextract extract --input /data/plans --output out/extractions.jsonl --batch
```

The images under `samples/` are synthetic (drawn by
`scripts/make_samples.py`, CC0); see
[`samples/README.md`](samples/README.md) for pointing the CLI at your own.

## Pipeline

```
   <root>/<project>/*.png  or  manifest.jsonl
                  │
                  ▼
   extract ───────────────────► output/extractions.jsonl   one row per image
   (tool | json mode;                 │
    sync | batch)                     ├─► classify   (optional sidecar: sheet type,
                                      │               unit code in image vs filename)
                                      ▼
   aggregate ─────────────────► output/projects.jsonl      one row per project
   (enrich · collapse ·               │
    merge unit numbers)               ├─► dq-profile / dq-clean
                                      └─► sync-pg
```

Full description, record shapes and flag reference: [docs/PIPELINE.md](docs/PIPELINE.md).
What did not work and why: [docs/LESSONS.md](docs/LESSONS.md).

## Cost

Order of magnitude with a Sonnet-class model:

| | per image | 30,000 images |
|---|---|---|
| sync | ~$0.022 | ~$660 |
| `--batch` | ~$0.011 | ~$330 |

Pilot on a few projects first (`--limit 3`), look at `projects.jsonl`, then
batch the rest.

## Commands

| command | what it does | API cost |
|---|---|---|
| `extract` | one structured record per image; sync or batch | yes |
| `aggregate` | per-project rollup with provenance and confidence | no |
| `classify` | sheet-type audit; where the unit code lives | yes |
| `try-one` / `try-crop` | single-image HTML verifiers | yes |
| `try-norm` | background normalisation / watermark removal HTML | no |
| `dq-profile` / `dq-clean` | data-quality report / idempotent cleaning | no |
| `sync-pg` | upsert projects into Postgres | no |

`fpextract <command> --help` documents every flag.

## Development

```bash
pip install -e '.[dev,cv]'
ruff check . && pytest -q
python scripts/make_samples.py   # regenerate the synthetic samples
```

Tests run offline; the LLM client is faked.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE). Use for personal, research,
educational and other noncommercial purposes is free. **Commercial use
requires a separate licence** — contact the author. Attribution is
required: keep the `Required Notice` line and the [NOTICE](NOTICE) file
with any copy. The synthetic samples are CC0.

Copyright 2026 Mr.Cyprinus (https://github.com/Cyprinus12138)
