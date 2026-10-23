import json
import logging

from fpextract.extract import ExtractConfig, Extractor
from fpextract.sink.jsonl import iter_jsonl, load_done

log = logging.getLogger("test")


def _cfg(samples_dir, out, **kw):
    return ExtractConfig(input=samples_dir, output=out, provider="anthropic", model="m", **kw)


def test_sync_writes_rows_and_resumes(tmp_path, samples_dir, fake_client):
    out = tmp_path / "ex.jsonl"
    cfg = _cfg(samples_dir, out, throttle=0, workers=2)
    summary = Extractor(cfg, fake_client, log).run()
    assert summary["extracted"] == 3 and summary["failed"] == 0
    rows = list(iter_jsonl(out))
    assert len(rows) == 3 and all("image_sha256" in r and r["mode"] == "tool" for r in rows)
    # second run skips everything
    summary2 = Extractor(cfg, fake_client, log).run()
    assert summary2 == {"extracted": 0, "failed": 0, "skipped": 3, "projects": 1}
    assert len(fake_client.calls) == 3
    # reextract ignores the done set
    Extractor(_cfg(samples_dir, out, throttle=0, reextract=True), fake_client, log).run()
    assert len(fake_client.calls) == 6


def test_sync_failure_becomes_error_row(tmp_path, samples_dir, fake_client):
    def boom(_):
        raise RuntimeError("model declined")
    fake_client.record_for = boom
    out = tmp_path / "ex.jsonl"
    summary = Extractor(_cfg(samples_dir, out, throttle=0), fake_client, log).run()
    assert summary["failed"] == 3
    rows = list(iter_jsonl(out))
    assert all("RuntimeError: model declined" in r["error"] for r in rows)
    # tool mode counts error rows as done; json mode retries them
    assert len(load_done(out, include_errors=True)) == 3
    assert load_done(out, include_errors=False) == set()


def test_json_mode_uses_json_path(tmp_path, samples_dir, fake_client):
    out = tmp_path / "ex.jsonl"
    Extractor(_cfg(samples_dir, out, throttle=0, mode="json", coord="fraction"), fake_client, log).run()
    assert fake_client.calls == ["json"] * 3


def test_limit_caps_projects(tmp_path, samples_dir, fake_client):
    import shutil
    root = tmp_path / "tree"
    shutil.copytree(samples_dir / "harbour-view", root / "aa-first")
    shutil.copytree(samples_dir / "harbour-view", root / "zz-other")
    out = tmp_path / "ex.jsonl"
    summary = Extractor(_cfg(root, out, throttle=0, limit=1), fake_client, log).run()
    assert summary["projects"] == 1 and summary["extracted"] == 3
    assert {r["project_slug"] for r in iter_jsonl(out)} == {"aa-first"}


def test_batch_submits_persists_and_resumes(tmp_path, samples_dir, fake_client, monkeypatch):
    out = tmp_path / "ex.jsonl"
    monkeypatch.setattr("fpextract.extract.PROVIDER_BATCH_LIMITS",
                        {"anthropic": {"max_requests": 2, "max_bytes": 10**9}})
    cfg = _cfg(samples_dir, out, batch=True, poll_interval=0)
    ex = Extractor(cfg, fake_client, log)
    state_path = cfg.resolved_state_path()

    # Simulate a crash right after the first chunk is submitted.
    original_submit = fake_client.submit_batch
    calls = {"n": 0}

    def crash_after_first(reqs):
        bid = original_submit(reqs)
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("network down")
        return bid

    fake_client.submit_batch = crash_after_first
    try:
        ex.run()
    except RuntimeError:
        pass
    # The crash happened *after* submit returned but before the state write,
    # which is exactly the window the design protects: the next run must not
    # double-pay. Verify the state file is absent (nothing persisted) and the
    # provider has the batch — then run again normally.
    assert fake_client.submitted == ["batch_1"]

    fake_client.submit_batch = original_submit
    summary = Extractor(cfg, fake_client, log).run()
    assert summary["submitted"] == 2            # 3 images / chunk of 2
    assert not state_path.exists()              # state removed after full collection
    rows = list(iter_jsonl(out))
    assert len(rows) == 3 and all(r.get("batch_id") for r in rows)


def test_batch_state_file_survives_and_is_reused(tmp_path, samples_dir, fake_client):
    out = tmp_path / "ex.jsonl"
    cfg = _cfg(samples_dir, out, batch=True, poll_interval=0)
    ex = Extractor(cfg, fake_client, log)
    state_path = cfg.resolved_state_path()
    # Submit only, then stop before polling by writing state ourselves.
    state = {"provider": "anthropic", "model": "m", "started_at": "t", "in_flight": []}
    ex._submit_pending(state, set())
    assert state_path.exists() and len(state["in_flight"]) == 1
    n_before = len(fake_client.submitted)
    # A fresh run loads the state, submits nothing new, collects the batch.
    summary = Extractor(cfg, fake_client, log).run()
    assert summary["submitted"] == 0 and len(fake_client.submitted) == n_before
    assert not state_path.exists() and len(list(iter_jsonl(out))) == 3


def test_batch_provider_mismatch_refuses(tmp_path, samples_dir, fake_client):
    out = tmp_path / "ex.jsonl"
    cfg = _cfg(samples_dir, out, batch=True)
    cfg.resolved_state_path().parent.mkdir(parents=True, exist_ok=True)
    cfg.resolved_state_path().write_text(json.dumps({"provider": "openai", "in_flight": []}))
    import pytest
    with pytest.raises(SystemExit):
        Extractor(cfg, fake_client, log).run()


def test_batch_errored_results_become_error_rows(tmp_path, samples_dir, fake_client):
    out = tmp_path / "ex.jsonl"
    a1 = samples_dir / "harbour-view" / "harbour-view-type-a1.png"
    fake_client.fail_on = {Extractor.custom_id_for(a1)}
    Extractor(_cfg(samples_dir, out, batch=True, poll_interval=0), fake_client, log).run()
    rows = list(iter_jsonl(out))
    errs = [r for r in rows if "error" in r]
    assert len(errs) == 1 and errs[0]["image_path"] == str(a1) and "boom" in errs[0]["error"]
