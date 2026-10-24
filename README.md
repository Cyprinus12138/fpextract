# fpextract

**English** · [简体中文](README.zh-CN.md)

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
multi-storey units, property type). Aggregation and data-quality processing
run locally and can be repeated without further LLM calls.

## Features

- **Provider-agnostic.** Anthropic, OpenAI, and any OpenAI-compatible
  endpoint (vLLM, Ollama, LM Studio, DashScope, Azure). Two prompting
  strategies: **tool** mode forces a call to the extraction tool via
  `tool_choice`, with an extraction schema but no strict mode; schema
  compliance depends on the model and endpoint. A plain-**json** prompt
  supports VLMs without function calling (Qwen3-VL, Qwen2.5-VL, LLaVA, …),
  with chain-of-thought stripping and per-model bounding-box normalisation.
- **Batch API with crash-safe resume.** Half price, auto-chunked to provider
  limits, every `batch_id` persisted by atomic rename the moment it is
  returned. Re-run the same command to resume batches recorded in the state
  file without re-submitting them. A failure between provider acceptance and
  saving the ID can leave a submitted batch untracked.
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

## Reviewing extractions in ProProp

In ProProp, this pipeline's output feeds the operations console's Data Studio, where a person reviews the projects the extractor and the reconciler flag. The floor plans are blurred in these shots: they are developers' brochure images, and this repository ships no third-party imagery.

<table>
<tr><td width="33%" valign="top"><a href="docs/images/proprop-data-studio.png"><img src="docs/images/proprop-data-studio.png" width="270" alt="Data Studio: flagged projects, lowest confidence first"></a><br><sub><b>Data Studio.</b> Projects flagged for a human pass, lowest confidence first.</sub></td><td width="33%" valign="top"><a href="docs/images/proprop-reconcile.png"><img src="docs/images/proprop-reconcile.png" width="270" alt="A project's source sheet beside its extracted units"></a><br><sub><b>One project.</b> The source sheet beside its units: area, bedrooms and bathrooms, storeys.</sub></td><td width="33%" valign="top"><a href="docs/images/proprop-room-editor.png"><img src="docs/images/proprop-room-editor.png" width="270" alt="Correcting room boxes by hand"></a><br><sub><b>Room editor.</b> Redrawing a unit's room boxes, with a required reason for the audit log.</sub></td></tr>
</table>

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

Tests run offline with a test double for the LLM client.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE). Use for personal, research,
educational and other noncommercial purposes is free. **Commercial use
requires a separate licence** — contact the author. Attribution is
required: keep the `Required Notice` line and the [NOTICE](NOTICE) file
with any copy. The synthetic samples are CC0.

Copyright 2026 Mr.Cyprinus (https://github.com/Cyprinus12138)
