from lib.dataset import Case
from lib.language import Language
from lib.query import build_query


def _case(*, facts: str = "", considerations: str = "") -> Case:
    return Case(
        decision_id="test",
        year=2020,
        language=Language.DE,
        facts=facts,
        considerations=considerations,
        rulings="",
        chamber="CH_BGer_001",
        region="Federation",
        origin_court=None,
        origin_canton=None,
        origin_chamber=None,
        law_area=None,
        law_sub_area=None,
        cited_rulings=[],
        laws=[],
    )


def test_concatenates_masked_facts_and_considerations():
    case = _case(facts="Sachverhalt A.", considerations="Erwägungen 1.")
    assert build_query(case=case) == "Sachverhalt A. Erwägungen 1."


def test_masks_citations_in_both_fields():
    # real example: case 048eb2ad-2de4-4b56-9739-e246e4490c5e
    case = _case(
        facts="Ein Schriftenwechsel wurde nicht angeordnet (Art. 129 Abs. 3 und Art. 127 BGG).",
        considerations="Eine Parteientschädigung ist nicht zuzusprechen (Art. 68 Abs. 1-3 BGG).",
    )
    query = build_query(case=case)
    assert "Art." not in query
    assert "BGG" not in query
    assert query.count("[LAW]") == 3


def test_empty_facts_has_no_leading_whitespace():
    case = _case(facts="", considerations="Erwägungen 1.")
    assert build_query(case=case) == "Erwägungen 1."


def test_empty_considerations_has_no_trailing_whitespace():
    case = _case(facts="Sachverhalt A.", considerations="")
    assert build_query(case=case) == "Sachverhalt A."


def test_both_empty():
    assert build_query(case=_case()) == ""
