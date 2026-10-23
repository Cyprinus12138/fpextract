# Lessons

Things that did not work, or worked differently than expected, while this
pipeline ran over roughly 30,000 marketing floor plans. Kept so the next
person does not repeat the experiments.

## Segmentation networks do not transfer to marketing drawings

The first plan was a full vectorisation path: a CubiCasa5k-style
segmentation model, a DSL for walls/openings/rooms, deterministic SVG
re-rendering and a 3D extrusion. It was parked. CubiCasa5k is trained on
scanned European floor plans rasterised from SVG; condominium marketing
sheets are a different domain (branded backgrounds, decorative furniture,
watermarks, multiple units per sheet, stack tables). Zero-shot wall F1 was
not usable and the per-sheet variety made fine-tuning uneconomical for
the actual product need, which was *structured attributes*, not geometry.
The research side of that question lives in
[fpvec-lab](https://github.com/Cyprinus12138/fpvec-lab).

The LLM extraction plus image normalisation covered the user-facing value:
rooms, areas, bedrooms, facing, unit-number maps, and a clean crop per unit.

## Tool-forced schemas beat prompting for JSON

With a schema-forced tool call the model cannot return malformed output,
and `additionalProperties: false` stops field drift. Plain-JSON prompting
is still needed for VLMs without function calling (Qwen-VL on vLLM), where
three failure modes had to be handled explicitly: `<think>…</think>`
chain-of-thought in the reply, the answer landing in `reasoning_content`
with an empty `content`, and truncation when the thinking budget eats the
output budget (hence `--max-tokens 16384`, `--no-think`, and retries).

## Every model has its own box coordinate system

Claude follows an instruction to emit fractions. Qwen2.5-VL emits 0–1000
integers. Qwen3-VL emits absolute pixels **of its internally smart-resized
image**, so the denominator has to be recomputed from the sent size with
the same resize rule. Normalising all three to `crop_frac` at the client
boundary kept everything downstream model-agnostic.

## The picture is not the only source of truth

The vision model cannot see the filename. For projects whose unit codes
exist only in filenames (`1-Bedroom-a4-b1.jpg`) it either returns a
generic descriptor ("1-Bedroom"), a room word ("MASTER BEDROOM"), or one
label repeated across distinct units. Two mitigations, both cheap:

- a sidecar pass that sees image *and* filename (`fpextract classify`);
- a cleaning rule that recovers the code from the filename when every
  source filename agrees (R7).

Likewise, property type (condo vs landed vs industrial) is often invisible
in a single sheet but obvious from a page caption or the project
description; the aggregate stage therefore takes project-level hints and
overrides per-record votes when they are confident.

## Stack charts masquerade as units

A storey × stack table lists every unit number with its type. Read as a
unit plan, it produces one "unit" per storey row with identical specs and a
numeric `floor_label` ("22", "23", …). This was the single largest DQ
issue by volume. Fixes: treat purely numeric floor labels as non-storeys,
dedupe identical-spec duplicates (R2), and keep stack tables out of
`plan_types` via the sidecar classification while still harvesting their
`unit_number_map`.

## Multi-storey detection needs a keyword gate

Collapsing on any `floor_label` difference falsely summed bedrooms across
*blocks* ("BLK 12", "Tower A", "Stack B1"). Only labels that contain a
storey keyword (ground, upper, lower, level N, Nth storey, attic, …) are
treated as storeys of one unit.

## Watermarks: three approaches, three trade-offs

- **SLBR end-to-end** removes the mark but runs at 512×512, so the output
  is blurry. Using SLBR only for the *mask* and inpainting at native
  resolution keeps the rest of the image pixel-perfect.
- **Mask + OpenCV TELEA** is sharp but cannot reconstruct line work under
  the mark; it only smears neighbours.
- **Mask + LaMa** reconstructs lines much better. IOPaint pinned an old
  torch and failed on recent CUDA wheels; `simple-lama-inpainting` had a
  loose constraint and worked.
- For a **known tiled text watermark**, a weight-free classical pipeline
  (auto-extracted template → match → grid projection → letter stencil →
  inpaint) was good enough and runs on CPU; one robustness fix mattered —
  choose the *tallest* high-density band for the template, not the first,
  or a single noise pixel short-circuits the template to 1–2 px.

## Batch APIs are worth the plumbing

Half price is the headline, but the operational property that mattered was
crash safety: persist the `batch_id` atomically before doing anything else,
and never re-submit. A 30K-image run spans hours and will be interrupted.

## Costs

Pilot on three projects before committing: a prompt change that looks fine
on one sheet can double the output tokens on a dense stack table. Sonnet-class
models landed at roughly 5,500 tokens per image.
