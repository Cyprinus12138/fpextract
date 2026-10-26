# Pipeline

`fpextract` turns a folder of floor-plan images into two queryable layers:
one record per **image** and one record per **project**. Aggregation and data
quality processing run locally without further LLM calls. The optional
`classify` and `try-crop` commands make additional LLM calls.

```
   <root>/<project>/*.png   or   manifest.jsonl           (input contract)
                       │
                       ▼
   Stage 1  fpextract extract                 vision LLM, ~$0.01–0.02 / image
            tool mode (schema-forced call)    sync  : one call per image, thread pool
            or json mode (plain JSON)         batch : provider batch API, half price,
                                                      auto-chunked, crash-safe resume
                       │
                       ▼
            output/extractions.jsonl          one row per IMAGE, append-safe:
                                              classification, confidence, units[]
                                              (rooms, areas, facing, crop_frac),
                                              unit_number_map, image_sha256
                       │
          ┌────────────┼──────────────────────────┐
          ▼            ▼                          ▼
   fpextract classify  fpextract aggregate   fpextract try-crop / try-norm
   (optional sidecar:  (local rollup)        (visual QA, CV refinement)
    sheet type, where
    the unit code is)      │
                           ▼
            output/projects.jsonl             one row per PROJECT:
                                              plan_types[] with provenance + confidence,
                                              unit_number_map, by_unit_type,
                                              unit_number_evidence, conflicts,
                                              multi-storey units, property type
                       │
          ┌────────────┼──────────────┐
          ▼            ▼              ▼
   dq-profile      dq-clean        sync-pg
   (markdown       (idempotent     (Postgres parent + child tables,
    report)         rules R1–R7)    atomic per project)
```

## Input contract

Two layouts are accepted wherever `--input` is taken (see
[`samples/README.md`](../samples/README.md)):

- a **directory tree** `<root>/<project>/*.{jpg,jpeg,png,webp}`; the folder
  name is the project slug;
- a **manifest JSONL**, one object per project with `project`, `images[]`
  and optional `title`, `url`, `description` (free text; keywords such as
  *bungalow* or *warehouse* drive property-type inference) and
  `property_type` (an explicit hint that wins over inference).

## Stage 1 — extract

```bash
fpextract extract --input samples --output out/extractions.jsonl --limit 1    # pilot
fpextract extract --input /data/plans --output out/extractions.jsonl --batch  # full run
```

The model classifies each sheet (`unit_plan`, `hybrid`, `site_map`,
`facility_plan`, `other`), returns one `units[]` entry per distinct unit
type with rooms, areas, facing and a `crop_frac` bounding box, and a
project-level `unit_number_map` whenever unit numbers are printed anywhere
on the sheet (cell labels, supplementary lists, stack-view tables), with
ranges expanded to one key per unit. The prompt is in
`src/fpextract/llm/prompts/extract.md`, the schema in
`src/fpextract/llm/schema/extraction.schema.json`.

### Prompting strategies

| `--mode` | How | Works with | Batch |
|---|---|---|---|
| `tool` (default) | forces a call to the extraction tool via `tool_choice` (no strict mode), so schema compliance depends on the model and endpoint | Anthropic, OpenAI, any OpenAI-compatible server with function calling | yes |
| `json` | plain-JSON prompt, tolerant parser (`<think>` blocks, code fences, reasoning fields), retry on truncation | Qwen3-VL / Qwen2.5-VL (vLLM, DashScope), LLaVA, InternVL, Ollama models without tools | no |

In `json` mode pass `--coord` to say which coordinate system the model
emits: `0-1000` (Qwen2.5-style normalised integers; also DashScope),
`qwen3` (absolute pixels of Qwen3-VL's smart-resized image), or `fraction`.
Boxes are converted to `crop_frac` so downstream code never sees the
difference.

```bash
# Qwen3-VL via DashScope's OpenAI-compatible endpoint
OPENAI_API_KEY=sk-... fpextract extract --input /data/plans --provider openai \
  --base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
  --model qwen3-vl-235b-a22b-instruct --mode json --coord 0-1000 --workers 4

# local vLLM Qwen3-VL-8B
fpextract extract --input /data/plans --provider openai --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct --mode json --coord qwen3 --no-think
```

### Sync vs batch

| | sync (default) | `--batch` |
|---|---|---|
| cost | list price | **50% off** (provider batch pricing) |
| latency | seconds per image | within 24 h, usually minutes |
| use for | prompt / model iteration, a few projects | the full corpus |
| resumable | yes: images already in the output are skipped | yes: state file |
| providers | anthropic, openai, any OpenAI-compatible | anthropic, openai |

Batch mode chunks the workload to each provider's per-batch limits
(Anthropic 100K requests / 256 MB, OpenAI 50K / 100 MB, with margin) and
submits chunk by chunk. **Every `batch_id` is written to
`.batch_state.json` via temp-file + atomic rename the moment `submit`
returns**, with no other I/O in between. Consequences:

- re-running the same command resumes batches recorded in the state file
  and submits images that are neither finished nor already in flight;
- recorded batches are not re-submitted. A failure after the provider accepts
  a batch but before its ID is saved can leave the batch untracked; atomic
  rename protects the state file from partial writes, not this submission gap;
- polling and collection errors are non-fatal and retried next cycle; the
  state file is only updated after a batch is fully collected;
- the state file refuses to load under a different `--provider`.

### Cost (order of magnitude)

```
~5,500 tokens / image avg   →  ~$0.022 sync, ~$0.011 batch   (Sonnet-class model)
30,000 images               →  ~$660 sync,  ~$330 batch
```

Pilot first (`--limit 3`), inspect `projects.jsonl`, then batch the rest.

## Stage 2 — aggregate

```bash
fpextract aggregate --extractions out/extractions.jsonl --input /data/plans --output out/projects.jsonl
fpextract aggregate ... --sheet-titles out/titles.jsonl     # optional sidecar
```

1. **Enrich** (optional). A per-image *sheet-title sidecar* carries ground
   truth the picture may not: `classification`, `property_type`,
   `plan_types[]`, `storey_label`. Site plans, stack charts and covers are
   dropped; a `per_unit_plan` title's `plan_types[]` overrides the picture's
   `unit_type` (1×M broadcast, N×N by document order, N×M>N top-N by area,
   N×M<N cycled). Without a sidecar every image passes through unchanged.
   Produce the sidecar however you like — a caption scraper, a spreadsheet,
   `fpextract classify` output mapped onto the four fields.
2. **Collapse** per `(project, unit_type)`. A storey signal (title
   `storey_label`, or the picture's `floor_label` when it contains a storey
   keyword) with two or more distinct values marks a **multi-storey unit**:
   measurements are summed, `storeys[]` keeps the per-level breakdown.
   Same-storey duplicates are consensus-merged (mode for counts and facing,
   median for areas, union for rooms). Each unit gets a `confidence`: best
   image confidence + agreement bonus − a penalty per disagreeing field.
3. **Unit numbers** are unioned across images; when images disagree the
   most-voted type wins (tie-break: image confidence). The flat
   `unit_number_map` stays clean for lookups; `unit_number_evidence` and
   `conflicts` carry the provenance.
4. **Property type** per project: an explicit `property_type` in the
   manifest wins, then keywords in `description`, then a majority of
   address-shaped unit types (landed houses), else the per-unit vote.

Per-project record, abridged:

```jsonc
{
  "project_slug": "harbour-view", "project_title": "Harbour View",
  "n_source_images": 3, "n_physical_units": 2, "n_multi_storey_units": 0,
  "dominant_property_type": "condo_apartment", "property_type_source": "llm_voted",
  "confidence": 0.86,
  "plan_types": [
    {"unit_type": "B2", "classification": "per_unit_plan", "n_storeys": 1,
     "bedrooms": 3, "bathrooms": 2, "total_area_sqm": 96, "facing": "N",
     "rooms": [...], "confidence": 0.9, "field_disagreements": [],
     "source_image_sha256s": ["…"], "source_image_basenames": ["harbour-view-type-b2.png"]}
  ],
  "unit_number_map": {"#02-01": "A1", "#02-02": "B2"},
  "by_unit_type": {"A1": ["#02-01"], "B2": ["#02-02"]},
  "unit_number_evidence": {"#02-01": {"unit_type": "A1", "confidence": 0.9,
                                       "source_count": 2, "conflicting_candidates": []}},
  "conflicts": []
}
```

## Classification sidecar

```bash
fpextract classify --extractions out/extractions.jsonl --output out/classifications.jsonl
```

Audits each unique image (content-addressed by `image_sha256`): which of
`per_unit_plan | stack_chart | site_plan | floor_plate | diagrammatic_chart |
brochure_cover | other` it is, whether a unit code is printed on the sheet
or only encoded in the filename, and the canonical code. The DQ cleaner's
rule R7 uses the same filename parser to recover codes such as
`1-Bedroom-a4-b1.jpg → a4b1`.

## Data quality

```bash
fpextract dq-profile --input out/projects.jsonl --output out/dq/report.md
fpextract dq-clean   --input out/projects.jsonl --output out/projects.cleaned.jsonl
```

The profiler is read-only and surfaces the long tail: room words and
generic descriptors used as unit types, numeric `floor_label`s from
stack-chart confusion, areas outside the plausible band for the bedroom
count, replicated records, phantom plan types, conflict leaderboard. The
cleaner applies seven targeted, idempotent rules (see the module docstring
in `dq/clean.py`) and writes a report of every row it touched.

## CV helpers (the `cv` extra)

- `fpextract try-crop IMAGE` — LLM box (orange) vs OpenCV-refined box
  (green). The refinement thresholds inside the padded LLM box (strategy
  chosen from the corner background tone), dilates to merge the drawing
  into one component, then measures the tight bounds of the undilated mask.
- `fpextract try-norm IMAGE [--tiled-watermark] [--slbr WEIGHTS]` — black-hat
  background normalisation (any tint → white), optional tiled text-watermark
  removal (template matched from the image itself, grid-projected, letter
  stencil inpainted), optional SLBR mask + TELEA inpaint.

## Postgres sink (the `pg` extra)

```bash
export DATABASE_URL=postgres://user:pass@host/db
fpextract sync-pg --input out/projects.jsonl --init-schema      # first time
fpextract sync-pg --input out/projects.jsonl --projects-only harbour-view
```

Schema in `src/fpextract/sink/schema.sql`: `floorplan_projects` (parent)
and `floorplan_units` (child, `UNIQUE (slug, unit_type)`). Each project is
written in one transaction.

## Flag reference

Run `fpextract <command> --help`; every flag is documented there.
