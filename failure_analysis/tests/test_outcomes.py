from failure_analysis.outcomes import CaseOutcome, evaluate_cases, worst_outcomes
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


def test_hits_is_the_intersection_of_retrieved_and_gold():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["a", "b"]),
        retrieved=["a", "x", "y"],
        gold=frozenset({"a", "b"}),
    )
    assert outcome.hits == frozenset({"a"})


def test_misses_is_gold_minus_retrieved():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["a", "b"]),
        retrieved=["a", "x", "y"],
        gold=frozenset({"a", "b"}),
    )
    assert outcome.misses == frozenset({"b"})


def test_recall_is_the_fraction_of_gold_hit():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["a", "b"]),
        retrieved=["a", "x", "y"],
        gold=frozenset({"a", "b"}),
    )
    assert outcome.recall == 0.5


def test_recall_of_a_complete_hit_is_one():
    outcome = CaseOutcome(
        case=_case(decision_id="c", cited_rulings=["a"]),
        retrieved=["a"],
        gold=frozenset({"a"}),
    )
    assert outcome.recall == 1.0


def test_evaluate_cases_pairs_each_case_with_its_own_retrieve_fn_result():
    cases = [
        _case(decision_id="c1", cited_rulings=["a"]),
        _case(decision_id="c2", cited_rulings=["b"]),
    ]
    retrieved_by_case = {"c1": ["a"], "c2": ["z"]}
    outcomes = evaluate_cases(
        retrieve_fn=lambda case: retrieved_by_case[case.decision_id], cases=cases
    )
    assert [o.recall for o in outcomes] == [1.0, 0.0]


def test_worst_outcomes_sorts_lowest_recall_first():
    good = CaseOutcome(
        case=_case(decision_id="good", cited_rulings=["a"]),
        retrieved=["a"],
        gold=frozenset({"a"}),
    )
    bad = CaseOutcome(
        case=_case(decision_id="bad", cited_rulings=["a"]),
        retrieved=[],
        gold=frozenset({"a"}),
    )
    assert worst_outcomes(outcomes=[good, bad], n=2) == [bad, good]


def test_worst_outcomes_respects_n():
    bad = CaseOutcome(
        case=_case(decision_id="bad", cited_rulings=["a"]),
        retrieved=[],
        gold=frozenset({"a"}),
    )
    good = CaseOutcome(
        case=_case(decision_id="good", cited_rulings=["a"]),
        retrieved=["a"],
        gold=frozenset({"a"}),
    )
    assert worst_outcomes(outcomes=[good, bad], n=1) == [bad]
