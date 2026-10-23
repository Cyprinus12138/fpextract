import pytest

from fpextract.llm.coords import bbox_to_crop_frac, normalise_record_bboxes, qwen_smart_resize


def test_fraction_clamps():
    assert bbox_to_crop_frac([-0.1, 0.2, 1.4, 0.9], "fraction", 100, 100) == [0.0, 0.2, 1.0, 0.9]


def test_0_1000():
    assert bbox_to_crop_frac([100, 200, 500, 900], "0-1000", 1, 1) == [0.1, 0.2, 0.5, 0.9]


def test_qwen3_uses_smart_resize():
    h, w = 2000, 3000
    sh, sw = qwen_smart_resize(h, w)
    assert sh % 28 == 0 and sw % 28 == 0 and sh * sw <= 1280 * 28 * 28
    frac = bbox_to_crop_frac([0, 0, sw, sh], "qwen3", h, w)
    assert frac == [0.0, 0.0, 1.0, 1.0]


def test_unknown_coord():
    with pytest.raises(ValueError):
        bbox_to_crop_frac([0, 0, 1, 1], "pixels", 1, 1)


def test_normalise_record_rewrites_bbox():
    rec = {"units": [{"unit_type": "A", "bbox": [0, 0, 500, 1000]}, {"unit_type": "B"}]}
    normalise_record_bboxes(rec, "0-1000", 10, 10)
    assert rec["units"][0]["crop_frac"] == [0.0, 0.0, 0.5, 1.0]
    assert "bbox" not in rec["units"][0] and "crop_frac" not in rec["units"][1]
