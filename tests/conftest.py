from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_make_samples():
    spec = importlib.util.spec_from_file_location("make_samples", ROOT / "scripts" / "make_samples.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_samples"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory) -> Path:
    """Synthetic sample sheets generated fresh into a temp dir."""
    out = tmp_path_factory.mktemp("samples")
    _load_make_samples().generate(out)
    return out


@pytest.fixture
def sample_images(samples_dir) -> list[Path]:
    return sorted((samples_dir / "harbour-view").glob("*.png"))


class FakeClient:
    """Stand-in for an LLM client: returns scripted records, supports batch."""

    provider = "fake"

    def __init__(self, record_for=None, fail_on: set[str] | None = None):
        self.record_for = record_for or (lambda name: {"classification": "unit_plan", "confidence": 0.9,
                                                       "unit_number_map": {}, "units": []})
        self.fail_on = fail_on or set()
        self.calls: list[str] = []
        self.batches: dict[str, list] = {}
        self.submitted: list[str] = []
        self.polls: dict[str, int] = {}
        self._n = 0

    # sync
    def extract_tool(self, image_bytes, media_type, model, system_prompt, tool_schema):
        self.calls.append("tool")
        return self.record_for("?"), {"input_tokens": 10, "output_tokens": 5}

    def complete_json_raw(self, image_bytes, media_type, model, system, prompt, max_tokens, no_think=False):
        self.calls.append("json")
        import json
        return json.dumps(self.record_for("?")), {"input_tokens": 1, "output_tokens": 1}

    def extract_json(self, image_bytes, media_type, model, prompt, coord="0-1000", **kw):
        self.calls.append("json")
        return self.record_for("?"), {"input_tokens": 1, "output_tokens": 1}

    # batch
    def submit_batch(self, requests):
        self._n += 1
        bid = f"batch_{self._n}"
        self.batches[bid] = list(requests)
        self.submitted.append(bid)
        self.polls[bid] = 0
        return bid

    def poll_batch(self, batch_id):
        from fpextract.llm.client import BatchStatus
        self.polls[batch_id] += 1
        done = self.polls[batch_id] >= 2
        return BatchStatus("completed" if done else "pending", "ended" if done else "in_progress", {})

    def fetch_batch_results(self, batch_id) -> Iterator:
        from fpextract.llm.client import BatchResult
        for r in self.batches[batch_id]:
            if r.custom_id in self.fail_on:
                yield BatchResult(r.custom_id, False, None, None, "errored: boom")
            else:
                yield BatchResult(r.custom_id, True, self.record_for(r.custom_id),
                                  {"input_tokens": 3, "output_tokens": 2}, None)


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()
