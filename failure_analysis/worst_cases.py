"""
Gathering a method's worst-recall cases and the ruling text they need to show.

Shared by the console (`failure_analysis.inspect_failures`) and HTML
(`failure_analysis.report`) failure-inspection tools, so both inspect the exact same
cases the exact same way.
"""

from dataclasses import dataclass, field
from pathlib import Path

from failure_analysis.outcomes import CaseOutcome, evaluate_cases, worst_outcomes
from lib.dataset import Split, iter_cases
from lib.dense.embedding_model import PASSAGE_PREFIX, chunk_for_model
from lib.retrieval_methods import Method, build_chunk_score_fn, build_retrieve_fn
from lib.rulings import Ruling, iter_rulings, list_ruling_ids
from lib.sampling import sample_items


@dataclass(frozen=True)
class WorstCases:
    """
    :ivar outcomes: The worst-recall case outcomes, worst first.
    :ivar rulings_by_id: Every gold ruling of any outcome, plus each outcome's
        top-ranked retrieved ruling, keyed by decision_id. A gold ruling_id absent from
        this dict doesn't exist in the ruling corpus at all (hence the ~90.7% resolution ceiling).
    :ivar val_case_count: How many validation cases were evaluated in total.
    :ivar chunk_scores: For methods that score ruling chunks: each case's (decision_id)
        score for every chunk of every ruling in `rulings_by_id` that it shows, in chunk
        order. Empty for methods that don't.
    :ivar chunks_by_id: For those methods: the chunks of every ruling in `rulings_by_id`
        (as the method split them), aligned with `chunk_scores`. Empty otherwise.
    """

    outcomes: list[CaseOutcome]
    rulings_by_id: dict[str, Ruling]
    val_case_count: int
    chunk_scores: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    chunks_by_id: dict[str, list[str]] = field(default_factory=dict)


def collect_worst_cases(
    *,
    method: Method,
    n: int | None,
    k: int,
    train_cases: int | None,
    val_queries: int | None,
    restrict_corpus: bool,
    seed: int,
    model_path: Path | None,
) -> WorstCases:
    """
    Evaluate `method` on a validation (sub)set and keep its `n` lowest-recall cases.

    :param method: Which retrieval method to inspect.
    :param n: How many of the worst cases to keep; all of them if None.
    :param k: How many results `method` retrieves per case.
    :param train_cases: Sample this many train cases (with citations) for the constant
        baseline; all of them if omitted. Ignored unless `method` is constant.
    :param val_queries: Sample this many validation queries (with citations) to search
        for failures in; all of them if omitted.
    :param restrict_corpus: Retrieve from a reduced corpus: only the rulings that the
        (sampled) train and validation cases cite.
    :param seed: The sampling seed.
    :param model_path: A fine-tuned model's directory to use for the dense method
        instead of the pretrained model (see `scripts.train`).
    :returns: The worst cases and the ruling text needed to display them.
    """
    train = sample_items(
        items=[c for c in iter_cases(split=Split.TRAIN) if c.cited_rulings],
        n=train_cases,
        seed=seed,
    )
    val_cases = sample_items(
        items=[c for c in iter_cases(split=Split.VALIDATION) if c.cited_rulings],
        n=val_queries,
        seed=seed,
    )
    ruling_ids: frozenset[str] | None = None
    if restrict_corpus:
        cited = {r for case in [*train, *val_cases] for r in case.cited_rulings}
        ruling_ids = frozenset(cited & set(list_ruling_ids()))

    retrieve_fn = build_retrieve_fn(
        method=method,
        train_cases=train,
        ruling_ids=ruling_ids,
        model_path=model_path,
        k=k,
    )
    outcomes = evaluate_cases(retrieve_fn=retrieve_fn, cases=val_cases)
    worst = worst_outcomes(outcomes=outcomes, n=n)

    shown_ruling_ids = frozenset(
        ruling_id
        for outcome in worst
        for ruling_id in [*outcome.gold, *outcome.retrieved[:1]]
    )
    rulings_by_id = {
        ruling.decision_id: ruling
        for ruling in iter_rulings(ruling_ids=shown_ruling_ids)
    }
    chunk_scores: dict[str, dict[str, list[float]]] = {}
    chunks_by_id: dict[str, list[str]] = {}
    chunk_score_fn = build_chunk_score_fn(
        method=method, ruling_ids=ruling_ids, model_path=model_path
    )
    if chunk_score_fn is not None:
        for outcome in worst:
            shown = [
                ruling_id
                for ruling_id in [*sorted(outcome.gold), *outcome.retrieved[:1]]
                if ruling_id in rulings_by_id
            ]
            chunk_scores[outcome.case.decision_id] = chunk_score_fn(outcome.case, shown)
        chunks_by_id = {
            ruling_id: chunk_for_model(
                text=ruling.considerations, prefix=PASSAGE_PREFIX
            )
            for ruling_id, ruling in rulings_by_id.items()
        }
    return WorstCases(
        outcomes=worst,
        rulings_by_id=rulings_by_id,
        val_case_count=len(val_cases),
        chunk_scores=chunk_scores,
        chunks_by_id=chunks_by_id,
    )
