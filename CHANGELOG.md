# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Provider-agnostic vision client with **tool** mode (schema-forced function
  call, Anthropic and OpenAI-style endpoints) and **json** mode (plain-JSON
  prompt for VLMs without function calling: Qwen3-VL / Qwen2.5-VL on vLLM or
  DashScope, LLaVA, InternVL, …), including `<think>` stripping and bounding-box
  coordinate normalisation (`fraction`, `0-1000`, `qwen3`).
- `extract` stage: sync runs with a thread pool, and batch runs on the provider
  batch API with auto-chunking to provider limits and a crash-safe state file
  (atomic rename right after every submit) so an interrupted run resumes
  recorded batches without re-submitting them. A failure before the batch ID
  is saved can still leave submitted work untracked.
- `aggregate` stage: optional sheet-title reconciliation, multi-storey unit
  collapse, unit-number vote merging with per-entity evidence, conflicts and
  confidence, and project-level property-type inference.
- Input contract independent of how images were obtained: a manifest JSONL or a directory
  tree.
- `classify` sidecar (sheet type; unit code in image vs filename).
- Data-quality profiler and idempotent cleaner (rules R1–R7).
- OpenCV crop snapping, background normalisation, tiled text-watermark
  removal, and an optional SLBR wrapper.
- HTML verifiers (`try-one`, `try-crop`, `try-norm`), JSONL and Postgres sinks.
- Synthetic CC0 sample sheets and their generator.

### Changed (vs. the pipeline this package was extracted from)
- Crop snapping measures the undilated content mask inside the merged
  component, so refined boxes are tight rather than inflated by the dilation
  kernel.
- Per-unit `confidence` and `field_disagreements` are carried through the
  multi-storey collapse instead of being dropped.
