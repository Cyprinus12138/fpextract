import jsonschema
import pytest

from fpextract.llm.client import load_extraction_schema, load_prompt


def test_schema_loads_without_dollar_keys():
    s = load_extraction_schema()
    assert not any(k.startswith("$") for k in s)
    assert s["required"] == ["classification", "confidence", "units"]
    assert "unit_number_map" in s["properties"]


def test_prompts_load():
    assert "record_extraction" in load_prompt("extract")
    assert "{COORD_INSTRUCTION}" in load_prompt("extract_json")


def test_good_record_validates():
    rec = {"classification": "unit_plan", "confidence": 0.9,
           "unit_number_map": {"#02-01": "A1"},
           "units": [{"unit_type": "A1", "bedrooms": 2, "bathrooms": 1, "total_area_sqm": 72.0,
                      "facing": "N", "rooms": [{"label": "LIVING", "area_sqm": 28.5}],
                      "crop_frac": [0.1, 0.1, 0.6, 0.6]}]}
    jsonschema.validate(rec, load_extraction_schema())


@pytest.mark.parametrize("bad", [
    {"classification": "bogus", "confidence": 0.5, "units": []},
    {"classification": "unit_plan", "confidence": 1.5, "units": []},
    {"classification": "unit_plan", "confidence": 0.5, "units": [{"unit_type": "A"}]},  # no crop_frac
    {"classification": "unit_plan", "confidence": 0.5, "units": [], "extra": 1},
])
def test_bad_records_rejected(bad):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, load_extraction_schema())
