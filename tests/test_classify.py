import json

import pytest

from fpextract.classify import derive_unit_code_from_filename, validate_classification
from fpextract.llm.client import parse_json_reply, strip_thinking


@pytest.mark.parametrize("stem,expected", [
    ("1-Bedroom-a4-b1-scaled", "a4b1"),
    ("3-bedroom-c3a-b-scaled", "c3ab"),
    ("2-bedroom-3-scaled", None),       # pure sequence number
    ("site-plan", None),
    ("Type-B2-Highrise", None),         # no descriptor prefix
    ("studio-lowerlevel", None),        # pure alphabetic tail
    ("penthouse-ph-2b", "ph2b"),
])
def test_derive_unit_code(stem, expected):
    assert derive_unit_code_from_filename(stem) == expected


def test_validate_classification_normalises_non_unit():
    out = validate_classification({"classification": "site_plan", "plan_type": "A1",
                                   "has_plan_type_in_picture": True})
    assert out == {"classification": "site_plan", "has_plan_type_in_picture": False,
                   "has_plan_type_in_filename": False, "plan_type": None}


def test_validate_classification_rejects_bad():
    with pytest.raises(ValueError):
        validate_classification({"classification": "nope"})
    with pytest.raises(ValueError):
        validate_classification({"classification": "per_unit_plan", "plan_type": 3})


def test_parse_json_reply_tolerates_noise():
    raw = "<think>hmm</think>```json\n{\"a\": 1}\n```\nDone."
    assert parse_json_reply(raw) == {"a": 1}
    assert strip_thinking("<think>x</think> y") == "y"
    with pytest.raises(json.JSONDecodeError):
        parse_json_reply("no json here")
