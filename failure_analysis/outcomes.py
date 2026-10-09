"""Per-case retrieval outcomes, for reading what a method got right and wrong."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from lib.dataset import Case


@dataclass(frozen=True)
class CaseOutcome:
    """
    One case's retrieval outcome against its gold `cited_rulings`.

    :ivar case: The query case.
    :ivar retrieved: The method's ranked results.
    :ivar gold: The case's cited rulings.
    """

    case: Case
    retrieved: list[str]
    gold: frozenset[str]

    @property
    def hits(self) -> frozenset[str]:
        """Gold rulings present in `retrieved`."""
        return self.gold & frozenset(self.retrieved)

    @property
    def misses(self) -> frozenset[str]:
        """Gold rulings absent from `retrieved`."""
        return self.gold - frozenset(self.retrieved)

    @property
    def recall(self) -> float:
        """The fraction of `gold` present in `retrieved`."""
        return len(self.hits) / len(self.gold)


def evaluate_cases(
    *, retrieve_fn: Callable[[Case], list[str]], cases: Iterable[Case]
) -> list[CaseOutcome]:
    """
    :param retrieve_fn: Given a case, returns its ranked retrieved ruling decision_ids.
    :param cases: Cases to evaluate - each must have non-empty cited_rulings.
    :returns: One outcome per case, in `cases` order.
    """
    return [
        CaseOutcome(
            case=case, retrieved=retrieve_fn(case), gold=frozenset(case.cited_rulings)
        )
        for case in cases
    ]


def worst_outcomes(
    *, outcomes: Iterable[CaseOutcome], n: int | None
) -> list[CaseOutcome]:
    """
    :param outcomes: Case outcomes to rank.
    :param n: How many of the lowest-recall outcomes to return; all if None.
    :returns: The `n` outcomes with the lowest recall, worst first. Ties keep
        `outcomes`' original relative order.
    """
    return sorted(outcomes, key=lambda o: o.recall)[:n]
