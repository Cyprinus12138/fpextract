import json

import pytest

from fpextract.inputs import iter_floor_plans, load_project_meta, parse_projects_only


def test_directory_tree(samples_dir):
    refs = list(iter_floor_plans(samples_dir))
    assert {r.project for r in refs} == {"harbour-view"}
    assert len(refs) == 3
    assert all(r.image_path.is_absolute() and r.image_path.exists() for r in refs)
    meta = load_project_meta(samples_dir)
    assert meta["harbour-view"].title == "harbour-view"


def test_manifest_relative_paths(samples_dir):
    manifest = samples_dir / "manifest.jsonl"
    refs = list(iter_floor_plans(manifest))
    assert len(refs) == 3 and all(r.image_path.exists() for r in refs)
    meta = load_project_meta(manifest)
    assert meta["harbour-view"].title == "Harbour View"
    assert "3 Bedroom" in meta["harbour-view"].description


def test_manifest_skips_missing_and_filters(samples_dir, tmp_path):
    manifest = tmp_path / "m.jsonl"
    rows = [
        {"project": "p1", "images": [str(samples_dir / "harbour-view" / "harbour-view-type-a1.png"), "/nope/x.png"]},
        {"project": "p2", "images": [str(samples_dir / "harbour-view" / "harbour-view-type-b2.png")]},
        {"images": ["orphan.png"]},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert [r.project for r in iter_floor_plans(manifest)] == ["p1", "p2"]
    assert [r.project for r in iter_floor_plans(manifest, {"p2"})] == ["p2"]


def test_missing_input_raises(tmp_path):
    with pytest.raises(SystemExit):
        list(iter_floor_plans(tmp_path / "missing"))


def test_parse_projects_only():
    assert parse_projects_only("a, b,,c") == {"a", "b", "c"}
    assert parse_projects_only("") is None
    assert parse_projects_only(None) is None
