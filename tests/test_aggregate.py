import json

from fpextract.aggregate import (
    aggregate,
    build_project_record,
    infer_project_property_type,
    looks_like_address,
    merge_unit_number_map,
)
from fpextract.enrich import coerce_meta, enrich_image
from fpextract.inputs import ProjectMeta
from fpextract.multistorey import collapse_unit, storey_signal, unit_confidence


def _rec(slug, path, units, unmap=None, conf=0.9):
    return {"project_slug": slug, "image_path": path, "image_sha256": path[-6:], "confidence": conf,
            "classification": "unit_plan", "unit_number_map": unmap or {}, "units": units,
            "extracted_at": "2026-01-01T00:00:00+00:00"}


def test_merge_unit_number_map_votes_and_conflicts():
    records = [
        _rec("p", "/a.png", [], {"#02-01": "A1", "#03-01": "A1"}, conf=0.9),
        _rec("p", "/b.png", [], {"#03-01": "B2"}, conf=0.95),
        _rec("p", "/c.png", [], {"#03-01": "A1"}, conf=0.5),
    ]
    flat, evidence, by_ut, conflicts = merge_unit_number_map(records)
    assert flat == {"#02-01": "A1", "#03-01": "A1"}           # 2 votes beat 1 higher-confidence vote
    assert by_ut == {"A1": ["#02-01", "#03-01"]}
    assert evidence["#03-01"]["source_count"] == 3
    assert evidence["#03-01"]["confidence"] == round(2 / 3 * 0.9, 3)
    assert conflicts == [{"unit_number": "#03-01", "winning": "A1",
                          "candidates": [{"unit_type": "A1", "source_count": 2},
                                         {"unit_type": "B2", "source_count": 1}]}]


def test_enrich_rules():
    rec = _rec("p", "/x.png", [{"unit_type": "TYPE A", "total_area_sqm": 50},
                               {"unit_type": "TYPE B", "total_area_sqm": 90}])
    assert all(r["_no_title"] for r in enrich_image(rec, None))
    assert enrich_image(rec, {"classification": "site_plan", "plan_types": []}) == []
    one = enrich_image(rec, {"classification": "per_unit_plan", "plan_types": ["C1"], "property_type": "x"})
    assert [r["unit_type"] for r in one] == ["C1", "C1"] and one[0]["_picture_unit_type"] == "TYPE A"
    top = enrich_image(rec, {"classification": "per_unit_plan", "plan_types": ["C1"], "property_type": "x"})
    assert len(top) == 2
    keep_top = enrich_image(_rec("p", "/y.png", rec["units"] + [{"unit_type": "C", "total_area_sqm": 10}]),
                            {"classification": "per_unit_plan", "plan_types": ["Q", "R"], "property_type": "x"})
    assert [r["_picture_unit_type"] for r in keep_top] == ["TYPE B", "TYPE A"]
    assert keep_top[0]["_match_method"] == "area_rank_NxM_keep_top"


def test_coerce_meta_slug_leak_and_site_titles():
    m = coerce_meta({"classification": "per_unit_plan", "plan_types": ["1-loft", "A1"],
                     "title": "Type A1", "property_type": "condo"}, "1-loft")
    assert m["plan_types"] == ["A1"]
    m2 = coerce_meta({"classification": "per_unit_plan", "plan_types": ["X"], "title": "Site Map",
                      "property_type": "condo"}, "p")
    assert m2["classification"] == "site_plan" and m2["plan_types"] == []


def test_storey_signal_filters_block_labels():
    assert storey_signal({"floor_label": "Upper Floor"}) == "Upper Floor"
    assert storey_signal({"floor_label": "BLK 12"}) is None
    assert storey_signal({"_title_storey_label": "Level 2", "floor_label": "Tower A"}) == "Level 2"


def test_collapse_multi_storey_sums_and_orders():
    rows = [
        {"project_slug": "p", "unit_type": "PH1", "_classification": "per_unit_plan", "_property_type": "condo",
         "floor_label": "Upper Floor", "bedrooms": 2, "bathrooms": 1, "total_area_sqm": 80.0,
         "_image_sha256": "s1", "_image_path": "/a/upper.png", "_image_confidence": 0.9, "rooms": [{"label": "BED"}]},
        {"project_slug": "p", "unit_type": "PH1", "_classification": "per_unit_plan", "_property_type": "condo",
         "floor_label": "Lower Floor", "bedrooms": 1, "bathrooms": 2, "total_area_sqm": 120.5,
         "_image_sha256": "s2", "_image_path": "/a/lower.png", "_image_confidence": 0.8, "rooms": [{"label": "LIVING"}]},
    ]
    u = collapse_unit(rows)
    assert u["n_storeys"] == 2
    assert [s["storey_label"] for s in u["storeys"]] == ["Lower Floor", "Upper Floor"]
    assert (u["bedrooms"], u["bathrooms"], u["total_area_sqm"]) == (3, 3, 200.5)
    assert u["source_image_basenames"] == ["upper.png", "lower.png"]
    assert u["n_source_records"] == 2


def test_unit_confidence_bonus_and_penalty():
    same = [{"_image_confidence": 0.8, "bedrooms": 2}, {"_image_confidence": 0.7, "bedrooms": 2}]
    assert unit_confidence(same) == (0.85, [])
    diff = [{"_image_confidence": 0.8, "bedrooms": 2}, {"_image_confidence": 0.7, "bedrooms": 3}]
    assert unit_confidence(diff) == (0.75, ["bedrooms"])


def test_property_type_inference():
    assert looks_like_address("121E Whitley Road")
    assert not looks_like_address("A1")
    units = [{"classification": "per_unit_plan", "unit_type": "12 Jalan Besar Lane"},
             {"classification": "per_unit_plan", "unit_type": "B2"}]
    assert infer_project_property_type(None, units) == "landed_house"
    assert infer_project_property_type(ProjectMeta("p", description="Warehouse units"), []) == "industrial"
    assert infer_project_property_type(ProjectMeta("p", property_type="commercial"), []) == "commercial"
    assert infer_project_property_type(ProjectMeta("p"), [{"classification": "per_unit_plan", "unit_type": "A"}]) is None


def test_build_project_record_shape():
    raw = [_rec("p", "/a.png", [{"unit_type": "A1", "bedrooms": 2, "crop_frac": [0, 0, 1, 1]}], {"#02-01": "A1"})]
    collapsed = [{"unit_type": "A1", "classification": "per_unit_plan", "property_type": "unknown",
                  "n_storeys": 1, "bedrooms": 2, "confidence": 0.9}]
    rec = build_project_record("p", raw, collapsed, ProjectMeta("p", title="P", url="u", description="bungalow"))
    assert rec["dominant_property_type"] == "landed_house" and rec["property_type_source"] == "project_inferred"
    assert rec["plan_types"][0]["property_type"] == "landed_house"
    assert rec["project_title"] == "P" and rec["n_physical_units"] == 1
    assert rec["by_unit_type"] == {"A1": ["#02-01"]}
    assert set(rec) >= {"confidence", "unit_number_evidence", "conflicts", "aggregated_at"}


def test_aggregate_end_to_end(tmp_path, samples_dir):
    a1 = samples_dir / "harbour-view" / "harbour-view-type-a1.png"
    rows = [_rec("harbour-view", str(a1), [{"unit_type": "A1", "bedrooms": 2, "crop_frac": [0, 0, 1, 1]}],
                 {"#02-01": "A1"}),
            {"project_slug": "harbour-view", "image_path": "/err.png", "error": "boom"}]
    ex = tmp_path / "ex.jsonl"
    ex.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    sidecar = tmp_path / "titles.jsonl"
    sidecar.write_text(json.dumps({"project": "harbour-view", "image": a1.name, "classification": "per_unit_plan",
                                   "plan_types": ["A1-G"], "property_type": "condo_apartment"}) + "\n")
    records, stats = aggregate(ex, input_source=samples_dir / "manifest.jsonl", sidecar=sidecar)
    assert stats["projects"] == 1 and records[0]["project_title"] == "Harbour View"
    assert [p["unit_type"] for p in records[0]["plan_types"]] == ["A1-G"]
    assert records[0]["dominant_property_type"] == "condo_apartment"
    assert records[0]["n_source_images"] == 1
