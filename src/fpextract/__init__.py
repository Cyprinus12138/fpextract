"""fpextract — structured floor-plan extraction with vision LLMs.

Pipeline: ``extract`` (one record per image) → ``aggregate`` (one record
per project, with provenance and confidence). Optional passes: ``classify``
(sheet-type audit), DQ profiling/cleaning, CV crop refinement and
watermark removal, and a Postgres sink.

Copyright 2026 Mr.Cyprinus. Licensed under the PolyForm Noncommercial
License 1.0.0 — see LICENSE.
"""

__version__ = "0.1.0"
