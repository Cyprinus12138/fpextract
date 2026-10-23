import json

from fpextract.dq.clean import area_is_egregious, clean_record, new_stats
from fpextract.dq.clean import run as clean_run
from fpextract.dq.profile import classify_floor_label, profile, render


def _proj(plans, **kw):
    return {"project_slug": "p", "plan_types": plans, "unit_number_map": {"#01-01": "A1"},
            "by_unit_type": {"A1": ["#01-01"]}, "confidence": 0.9, **kw}


def test_rules_r1_r3_r4_r5_r6():
    stats = new_stats()
    out = clean_record(_proj([
        {"unit_type": "MASTER BEDROOM", "bedrooms": 1},
        {"unit_type": "A1", "bedrooms": 0, "total_area_sqm": 0, "facing": "sideways"},
        {"unit_type": "2-Bedroom", "bedrooms": 2, "total_area_sqm": 2000},
    ]), stats)
    plans = out["plan_types"]
    assert [p["unit_type"] for p in plans] == ["A1", "2-Bedroom"]
    assert plans[0]["bedrooms"] is None and plans[0]["total_area_sqm"] is None and plans[0]["facing"] == "unknown"
    assert plans[1]["unit_type_is_generic"] is True and plans[1]["total_area_sqm"] is None
    assert {k: len(v) for k, v in stats.items()} == {
        "r7_filename_recovered": 0, "r1_dropped": 1, "r2_replicated_dedup": 0, "r3_zero_bed_nullified": 1,
        "r4_area_nullified": 2, "r5_generic_tagged": 1, "r6_facing_normalised": 1}


def test_r7_filename_recovery_needs_consensus():
    stats = new_stats()
    out = clean_record(_proj([
        {"unit_type": "1-Bedroom", "source_image_basenames": ["1-Bedroom-a4-b1-scaled.jpg", "1-bedroom-a4-b1.png"]},
        {"unit_type": "LIVING", "source_image_basenames": ["1-Bedroom-a4-b1.jpg", "1-Bedroom-b3.jpg"]},
        {"unit_type": "B2", "source_image_basenames": ["1-Bedroom-a4-b1.jpg"]},
    ]), stats)
    assert out["plan_types"][0]["unit_type"] == "a4b1" and out["plan_types"][0]["unit_type_source"] == "filename"
    assert [p["unit_type"] for p in out["plan_types"]] == ["a4b1", "B2"]  # LIVING dropped (no consensus)
    assert len(stats["r7_filename_recovered"]) == 1


def test_r2_dedupes_replicated_stack_rows():
    stats = new_stats()
    out = clean_record(_proj([
        {"unit_type": "A1", "bedrooms": 2, "bathrooms": 1, "total_area_sqm": 70, "floor_label": "22",
         "confidence": 0.8, "source_image_sha256s": ["s1"]},
        {"unit_type": "A1", "bedrooms": 2, "bathrooms": 1, "total_area_sqm": 70, "floor_label": "23",
         "confidence": 0.9, "source_image_sha256s": ["s2"]},
    ]), stats)
    assert len(out["plan_types"]) == 1
    w = out["plan_types"][0]
    assert w["confidence"] == 0.9 and w["floor_label"] is None and w["source_image_sha256s"] == ["s2", "s1"]


def test_clean_is_idempotent(tmp_path):
    src = tmp_path / "p.jsonl"
    src.write_text(json.dumps(_proj([{"unit_type": "A1", "bedrooms": 0, "facing": "up"}])) + "\n")
    once = tmp_path / "once.jsonl"
    clean_run(src, once, tmp_path / "r1.md")
    twice = tmp_path / "twice.jsonl"
    s = clean_run(once, twice, tmp_path / "r2.md")
    assert once.read_text() == twice.read_text()
    assert s["r3_zero_bed_nullified"] == 0 and s["r6_facing_normalised"] == 0


def test_area_is_egregious():
    assert area_is_egregious(2, 0)
    assert area_is_egregious(2, 300)
    assert not area_is_egregious(2, 100)
    assert not area_is_egregious(None, 5)


def test_profile_and_render():
    recs = [_proj([{"unit_type": "A1", "bedrooms": 2, "bathrooms": 5, "total_area_sqm": 70, "facing": "N",
                    "confidence": 0.9, "classification": "per_unit_plan", "n_storeys": 2},
                   {"unit_type": "KITCHEN", "bedrooms": 1}],
                  dominant_property_type="condo_apartment", conflicts=[{"unit_number": "x"}])]
    rep = profile(recs)
    assert rep["totals"]["n_plan_types"] == 2 and rep["unit_type"]["room_words"] == [("KITCHEN", 1)]
    assert rep["invariants"]["bath_gt_bed"] == [("p", "A1", 2, 5)]
    assert rep["anomalies"]["phantom_plan_types"] == [("p", "KITCHEN")]
    assert rep["v2"]["multi_storey_units"] == 1 and rep["conflicts"]["n_total"] == 1
    md = render(rep, "x.jsonl")
    assert "# Floor-plan data quality report" in md and "KITCHEN" in md
    assert classify_floor_label("22") == "numeric" and classify_floor_label("Upper Floor") == "storey_label"
