"""
Read a method's worst retrieval failures.

Shows enough text to see why they happened: the query, and each missed gold ruling's own
text (the qualitative half of error analysis, not the per-category breakdown).
"""

from pathlib import Path

import cyclopts

from failure_analysis.outcomes import CaseOutcome
from failure_analysis.worst_cases import collect_worst_cases
from lib.query import build_query
from lib.retrieval_methods import Method
from lib.rulings import Ruling


def _snippet(*, text: str, length: int) -> str:
    """
    :param text: The text to shorten.
    :param length: The maximum length to keep.
    :returns: `text`, truncated to `length` characters with a trailing ellipsis if it
        was longer.
    """
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= length else f"{collapsed[:length]}..."


def _print_outcome(
    *, outcome: CaseOutcome, rulings_by_id: dict[str, Ruling], snippet_length: int
) -> None:
    """
    :param outcome: The case outcome to print.
    :param rulings_by_id: Every ruling `outcome` might miss, keyed by decision_id.
    :param snippet_length: How many characters of each text snippet to show.
    """
    case = outcome.case
    print(
        f"\n{case.decision_id}  {case.language}  {case.law_area or '(no law_area)'}  "
        f"recall={outcome.recall:.2f}  ({len(outcome.hits)}/{len(outcome.gold)} hit)"
    )
    print(f"  query: {_snippet(text=build_query(case=case), length=snippet_length)}")
    for ruling_id in sorted(outcome.hits):
        print(f"  HIT   {ruling_id}")
    for ruling_id in sorted(outcome.misses):
        ruling = rulings_by_id.get(ruling_id)
        if ruling is None:
            # ~90.7% of cited_rulings resolve against the corpus - the
            # rest don't exist here at all, so no method could ever retrieve them.
            print(f"  MISS  {ruling_id}  (not in the ruling corpus - unretrievable)")
            continue
        print(f"  MISS  {ruling_id}  ({ruling.language})")
        print(f"        {_snippet(text=ruling.considerations, length=snippet_length)}")


def inspect_failures(
    *,
    method: Method,
    n: int = 10,
    k: int = 10,
    snippet_length: int = 300,
    train_cases: int | None = None,
    val_queries: int | None = None,
    restrict_corpus: bool = False,
    seed: int = 0,
    model_path: Path | None = None,
) -> None:
    """
    Print the `n` validation cases where `method` has the lowest Recall@k, with the
    query and every missed gold ruling's text, for reading why they failed.

    :param method: Which retrieval method to inspect.
    :param n: How many of the worst cases to print.
    :param k: How many results `method` retrieves per case.
    :param snippet_length: How many characters of each query/ruling text to show.
    :param train_cases: Sample this many train cases (with citations) for the constant
        baseline; all of them if omitted. Ignored unless `method` is constant.
    :param val_queries: Sample this many validation queries (with citations) to search
        for failures in; all of them if omitted.
    :param restrict_corpus: Retrieve from a reduced corpus: only the rulings that the
        (sampled) train and validation cases cite.
    :param seed: The sampling seed.
    :param model_path: A fine-tuned model's directory to use for the dense method
        instead of the pretrained model (see `scripts.train`).
    """
    worst_cases = collect_worst_cases(
        method=method,
        n=n,
        k=k,
        train_cases=train_cases,
        val_queries=val_queries,
        restrict_corpus=restrict_corpus,
        seed=seed,
        model_path=model_path,
    )
    print(
        f"{worst_cases.val_case_count} cases evaluated, showing the "
        f"{len(worst_cases.outcomes)} worst"
    )
    for outcome in worst_cases.outcomes:
        _print_outcome(
            outcome=outcome,
            rulings_by_id=worst_cases.rulings_by_id,
            snippet_length=snippet_length,
        )


if __name__ == "__main__":
    cyclopts.run(inspect_failures)
