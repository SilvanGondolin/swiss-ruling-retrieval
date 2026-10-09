import failure_analysis.inspect_failures as m
from failure_analysis.outcomes import CaseOutcome
from lib.dataset import Case
from lib.language import Language


def _case(*, decision_id: str, cited_rulings: list[str]) -> Case:
    return Case(
        decision_id=decision_id,
        year=2020,
        language=Language.DE,
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


def test_print_outcome_handles_a_gold_ruling_absent_from_the_corpus(capsys):
    # ~90.7% of cited_rulings resolve against rcds/swiss_leading_decisions; the rest - like "unresolvable" here - have no entry in rulings_by_id at all.
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["unresolvable"]),
        retrieved=[],
        gold=frozenset({"unresolvable"}),
    )
    m._print_outcome(outcome=outcome, rulings_by_id={}, snippet_length=300)
    assert "unresolvable" in capsys.readouterr().out
