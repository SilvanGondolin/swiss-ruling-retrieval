import dataclasses
import json
import math
import re

import failure_analysis.html_report as m
from failure_analysis.html_report import render_report
from failure_analysis.outcomes import CaseOutcome
from failure_analysis.worst_cases import WorstCases
from lib.dataset import Case
from lib.language import Language
from lib.lexical.bm25 import DocumentFrequencies
from lib.rulings import Ruling


def _case(
    *, decision_id: str, cited_rulings: list[str], language: Language = Language.DE
) -> Case:
    return Case(
        decision_id=decision_id,
        year=2020,
        language=language,
        facts="",
        considerations="",
        rulings="",
        chamber="CH_BGer_001",
        region="Federation",
        origin_court=None,
        origin_canton=None,
        origin_chamber=None,
        law_area=None,
        law_sub_area=None,
        cited_rulings=cited_rulings,
        laws=[],
    )


def _ruling(*, decision_id: str, considerations: str = "") -> Ruling:
    return Ruling(
        decision_id=decision_id,
        language_id=1,
        chamber_id=1,
        file_id=1,
        date=None,
        topic="",
        year=2020,
        language=Language.DE,
        facts="",
        facts_num_tokens_bert=0,
        facts_num_tokens_spacy=0,
        considerations=considerations,
        considerations_num_tokens_bert=0,
        considerations_num_tokens_spacy=0,
        rulings="",
        rulings_num_tokens_bert=0,
        rulings_num_tokens_spacy=0,
        chamber="",
        court="",
        canton="",
        region="",
        file_name="",
        html_url="",
        pdf_url="",
        file_number="",
    )


def _frequencies() -> DocumentFrequencies:
    return DocumentFrequencies(counts={}, totals={lang: 100 for lang in Language})


def _payload(html: str) -> dict:
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', html)
    assert match is not None
    return json.loads(match.group(1))


def _report(*, outcomes: list[CaseOutcome], rulings: list[Ruling]) -> str:
    return render_report(
        worst_cases=WorstCases(
            outcomes=outcomes,
            rulings_by_id={r.decision_id: r for r in rulings},
            val_case_count=len(outcomes),
        ),
        method_label="bm25",
        frequencies=_frequencies(),
    )


def test_render_report_has_one_clickable_row_per_case():
    outcomes = [
        CaseOutcome(
            case=_case(decision_id=f"c{i}", cited_rulings=["r"]),
            retrieved=["r"],
            gold=frozenset({"r"}),
        )
        for i in range(3)
    ]
    html = _report(outcomes=outcomes, rulings=[_ruling(decision_id="r")])
    assert html.count("class='case-row'") == 3
    assert html.count("onclick='toggleCase(this)'") == 3
    assert [c["options"][0]["id"] for c in _payload(html)["cases"]] == ["r"] * 3


def test_render_row_summarizes_recall_hits_and_top_prediction():
    outcome = CaseOutcome(
        case=_case(decision_id="c1", cited_rulings=["a", "b"]),
        retrieved=["a", "x"],
        gold=frozenset({"a", "b"}),
    )
    options = m._ruling_options(
        outcome=outcome,
        rulings_by_id={r: _ruling(decision_id=r) for r in ["a", "b"]},
    )
    row = m._render_row(index=0, outcome=outcome, options=options)
    assert "<td data-sort='0.5'>0.50</td>" in row
    assert "<td data-sort='1'>1</td><td data-sort='2'>2</td>" in row
    assert "<td data-sort='yes'>yes</td>" in row
    assert "data-statuses='hit miss'" in row


def test_render_report_escapes_text_from_the_data():
    outcome = CaseOutcome(
        case=_case(decision_id="<script>alert(1)</script>", cited_rulings=["r"]),
        retrieved=[],
        gold=frozenset({"r"}),
    )
    html = _report(
        outcomes=[outcome],
        rulings=[_ruling(decision_id="r", considerations="</script><b>bold</b>")],
    )
    assert "<script>alert(1)</script>" not in html
    assert "<b>bold</b>" not in html
    assert html.count("</script>") == 2  # only the data and code scripts' own
    assert _payload(html)["rulings"]["r"] == ["</script><b>bold</b>"]


def test_render_report_is_valid_enough_html_to_open():
    html = _report(outcomes=[], rulings=[])
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")


def test_render_report_offers_filters_by_language_and_ruling_status():
    html = _report(outcomes=[], rulings=[])
    assert 'id="min-idf"' in html
    assert 'id="ignore-numbers"' in html
    assert 'id="show-sequences"' in html
    assert 'id="min-seq"' in html
    assert '<select id="language"' in html
    assert '<select id="status"' in html
    for value in ["hit", "miss", "wrong"]:
        assert f'<option value="{value}">' in html


def test_render_report_embeds_each_ruling_text_once_for_many_cases():
    outcomes = [
        CaseOutcome(
            case=_case(decision_id=f"c{i}", cited_rulings=["r"]),
            retrieved=["r"],
            gold=frozenset({"r"}),
        )
        for i in range(2)
    ]
    html = _report(
        outcomes=outcomes,
        rulings=[_ruling(decision_id="r", considerations="unique text")],
    )
    assert html.count("unique text") == 1


def test_case_payload_marks_a_ruling_missing_from_the_corpus_unresolvable():
    outcome = CaseOutcome(
        case=_case(decision_id="c1", cited_rulings=["gone"]),
        retrieved=[],
        gold=frozenset({"gone"}),
    )
    [option] = m._case_payload(
        outcome=outcome,
        options=m._ruling_options(outcome=outcome, rulings_by_id={}),
        frequencies=_frequencies(),
        chunk_scores={},
    )["options"]
    assert option["hasRuling"] is False
    assert option["status"] == "unresolvable"
    assert option["label"] == "not in corpus"


def test_case_payload_lists_gold_rulings_for_tooltips_but_not_the_wrong_prediction():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["hit", "miss", "gone"]),
        retrieved=["wrong", "hit"],
        gold=frozenset({"hit", "miss", "gone"}),
    )
    payload = m._case_payload(
        outcome=outcome,
        options=m._ruling_options(
            outcome=outcome,
            rulings_by_id={r: _ruling(decision_id=r) for r in ["hit", "miss", "wrong"]},
        ),
        frequencies=_frequencies(),
        chunk_scores={},
    )
    assert [(g["id"], g["status"]) for g in payload["gold"]] == [
        ("hit", "hit"),
        ("miss", "miss"),
        ("gone", "unresolvable"),
    ]
    assert payload["citationClass"] == "mixed"


def test_case_payload_lists_the_words_query_and_ruling_share():
    case = dataclasses.replace(
        _case(decision_id="c1", cited_rulings=["r"]),
        considerations="Sicherheitshaft und Verwahrung",
    )
    outcome = CaseOutcome(case=case, retrieved=["r"], gold=frozenset({"r"}))
    ruling = _ruling(decision_id="r", considerations="Die Sicherheitshaft, Rückfall")
    [option] = m._case_payload(
        outcome=outcome,
        options=m._ruling_options(outcome=outcome, rulings_by_id={"r": ruling}),
        frequencies=DocumentFrequencies(
            counts={"sicherheitshaft": 4}, totals={Language.DE: 100}
        ),
        chunk_scores={},
    )["options"]
    assert list(option["sharedWords"]) == ["sicherheitshaft"]
    # a stem in 4 of 100 rulings
    assert option["sharedWords"]["sicherheitshaft"] == round(
        math.log(1 + 96.5 / 4.5), 2
    )


def test_shared_words_include_the_inflected_forms_of_either_text():
    ruling = _ruling(decision_id="r", considerations="Das Verfahren dauert.")
    shared = m._shared_words(
        query="Im Verfahren und in den Verfahren",
        query_language=Language.DE,
        ruling=ruling,
        frequencies=_frequencies(),
    )
    assert set(shared.idf_by_word) == {"verfahren"}
    assert len(shared.stems) == 1


def test_shared_words_ignore_stopwords_and_placeholders():
    ruling = _ruling(decision_id="r", considerations="der [LAW] [RULING] die")
    shared = m._shared_words(
        query="der [LAW] [RULING] die",
        query_language=Language.DE,
        ruling=ruling,
        frequencies=_frequencies(),
    )
    assert shared.stems == frozenset()


def test_citation_overall_class_is_all_hit_when_nothing_was_missed():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["a"]),
        retrieved=["a"],
        gold=frozenset({"a"}),
    )
    assert m._citation_overall_class(outcome=outcome) == "all-hit"


def test_citation_overall_class_is_all_miss_when_nothing_was_hit():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["a"]),
        retrieved=[],
        gold=frozenset({"a"}),
    )
    assert m._citation_overall_class(outcome=outcome) == "all-miss"


def test_citation_overall_class_is_mixed_with_both_hits_and_misses():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["a", "b"]),
        retrieved=["a"],
        gold=frozenset({"a", "b"}),
    )
    assert m._citation_overall_class(outcome=outcome) == "mixed"


def test_ruling_options_lists_hits_then_misses_then_unresolvable_then_wrong_top():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["hit", "miss", "gone"]),
        retrieved=["wrong", "hit"],
        gold=frozenset({"hit", "miss", "gone"}),
    )
    options = m._ruling_options(
        outcome=outcome,
        rulings_by_id={r: _ruling(decision_id=r) for r in ["hit", "miss", "wrong"]},
    )
    assert [(o.ruling_id, o.status) for o in options] == [
        ("hit", m.RulingStatus.HIT),
        ("miss", m.RulingStatus.MISS),
        ("gone", m.RulingStatus.UNRESOLVABLE),
        ("wrong", m.RulingStatus.WRONG),
    ]


def test_ruling_options_omits_the_top_prediction_when_it_is_cited():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["a"]),
        retrieved=["a", "x"],
        gold=frozenset({"a"}),
    )
    options = m._ruling_options(
        outcome=outcome, rulings_by_id={"a": _ruling(decision_id="a")}
    )
    assert [o.status for o in options] == [m.RulingStatus.HIT]


def test_ruling_option_name_prefers_the_file_name():
    ruling = dataclasses.replace(_ruling(decision_id="id"), file_name="BGE_1_I_1")
    named = m.RulingOption(ruling_id="id", ruling=ruling, status=m.RulingStatus.HIT)
    unnamed = m.RulingOption(ruling_id="id", ruling=None, status=m.RulingStatus.HIT)
    assert (named.name, unnamed.name) == ("BGE_1_I_1", "id")


def test_statuses_counts_an_unresolvable_ruling_as_a_miss():
    options = [
        m.RulingOption(ruling_id="a", ruling=None, status=m.RulingStatus.UNRESOLVABLE),
        m.RulingOption(ruling_id="b", ruling=None, status=m.RulingStatus.WRONG),
    ]
    assert m._statuses(options=options) == "miss wrong"


def test_case_payload_marks_the_best_scoring_chunk_of_each_ruling():
    outcome = CaseOutcome(
        case=_case(decision_id="c1", cited_rulings=["r"]),
        retrieved=["r"],
        gold=frozenset({"r"}),
    )
    [option] = m._case_payload(
        outcome=outcome,
        options=m._ruling_options(
            outcome=outcome, rulings_by_id={"r": _ruling(decision_id="r")}
        ),
        frequencies=_frequencies(),
        chunk_scores={"r": [1.00004, 7.5, 3.0]},
    )["options"]
    assert option["chunkScores"] == [1.0, 7.5, 3.0]
    assert option["bestChunk"] == 1


def test_case_payload_has_no_chunk_data_for_methods_without_chunk_scores():
    outcome = CaseOutcome(
        case=_case(decision_id="c1", cited_rulings=["r"]),
        retrieved=["r"],
        gold=frozenset({"r"}),
    )
    [option] = m._case_payload(
        outcome=outcome,
        options=m._ruling_options(
            outcome=outcome, rulings_by_id={"r": _ruling(decision_id="r")}
        ),
        frequencies=_frequencies(),
        chunk_scores={},
    )["options"]
    assert option["chunkScores"] is None
    assert option["bestChunk"] is None


def test_best_chunk_takes_the_first_of_tied_chunks():
    assert m._best_chunk([2.0, 5.0, 5.0]) == 1


def test_render_report_embeds_rulings_as_the_chunks_the_method_scored():
    outcome = CaseOutcome(
        case=_case(decision_id="c1", cited_rulings=["r"]),
        retrieved=["r"],
        gold=frozenset({"r"}),
    )
    worst_cases = WorstCases(
        outcomes=[outcome],
        rulings_by_id={"r": _ruling(decision_id="r", considerations="one two")},
        val_case_count=1,
        chunk_scores={"c1": {"r": [0.1, 0.9]}},
        chunks_by_id={"r": ["one", "two"]},
    )
    payload = _payload(
        render_report(
            worst_cases=worst_cases, method_label="bm25", frequencies=_frequencies()
        )
    )
    assert payload["rulings"]["r"] == ["one", "two"]
    assert payload["cases"][0]["options"][0]["bestChunk"] == 1


def test_script_has_no_string_literal_broken_across_lines():
    # _SCRIPT is a plain Python string, so a JS "\n" must be written "\\n" there: a raw
    # newline inside a JS string literal is a syntax error that kills the whole page
    for line in m._SCRIPT.splitlines():
        code = line.split("//")[0]
        assert code.count("'") % 2 == 0, line
