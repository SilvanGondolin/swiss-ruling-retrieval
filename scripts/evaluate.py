"""Evaluate the BM25 and zero-shot dense baselines against the validation split's
cited_rulings target, alongside the most-common-rulings constant baseline for
comparison."""

import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cyclopts
from tqdm import tqdm

from lib.dataset import Case, Split, iter_cases
from lib.metrics import mrr, top_k_ndcg, top_k_recall
from lib.retrieval_methods import Method, build_retrieve_fn
from lib.rulings import list_ruling_ids
from lib.sampling import sample_items

_ROW_LABELS = {
    Method.CONSTANT: "constant",
    Method.BM25: "BM25",
    Method.BM25_CHUNKED: "BM25-chunk",
    Method.IDF_OVERLAP: "IDF-overlap",
    Method.DENSE: "dense",
}


@dataclass(frozen=True)
class Scores:
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    ndcg_at_10: float
    mrr: float


@dataclass(frozen=True)
class _MetricColumn:
    """
    :ivar attribute: The `Scores` field the column shows.
    :ivar header: The column's header text.
    """

    attribute: str
    header: str


_METRIC_COLUMNS = [
    _MetricColumn(attribute="recall_at_10", header="Recall@10"),
    _MetricColumn(attribute="recall_at_50", header="Recall@50"),
    _MetricColumn(attribute="recall_at_100", header="Recall@100"),
    _MetricColumn(attribute="ndcg_at_10", header="nDCG@10"),
    _MetricColumn(attribute="mrr", header="MRR"),
]


def _evaluate(
    *, retrieve_fn: Callable[[Case], list[str]], cases: Iterable[Case]
) -> Scores:
    """
    Compute mean Recall@10/50/100, nDCG@10, and MRR over `cases`.

    :param retrieve_fn: Given a case, returns up to 100 ranked ruling decision_ids.
    :param cases: Cases to evaluate over - each must have non-empty cited_rulings.
    :returns: The five mean metric values.
    """
    recall_10: list[float] = []
    recall_50: list[float] = []
    recall_100: list[float] = []
    ndcg_10: list[float] = []
    mrrs: list[float] = []
    for case in cases:
        gold = set(case.cited_rulings)
        retrieved = retrieve_fn(case)
        recall_10.append(top_k_recall(retrieved=retrieved, gold=gold, k=10))
        recall_50.append(top_k_recall(retrieved=retrieved, gold=gold, k=50))
        recall_100.append(top_k_recall(retrieved=retrieved, gold=gold, k=100))
        ndcg_10.append(top_k_ndcg(retrieved=retrieved, gold=gold, k=10))
        mrrs.append(mrr(retrieved=retrieved, gold=gold))
    return Scores(
        recall_at_10=statistics.mean(recall_10),
        recall_at_50=statistics.mean(recall_50),
        recall_at_100=statistics.mean(recall_100),
        ndcg_at_10=statistics.mean(ndcg_10),
        mrr=statistics.mean(mrrs),
    )


def _print_row(*, label: str, scores: Scores) -> None:
    """
    :param label: The method name for the row's first column.
    :param scores: The row's metric values.
    """
    values = "".join(
        f"{getattr(scores, column.attribute):>11.3f}" for column in _METRIC_COLUMNS
    )
    print(f"{label:11}{values}")


def evaluate(
    *,
    methods: list[Method] | None = None,
    train_cases: int | None = None,
    val_queries: int | None = None,
    restrict_corpus: bool = False,
    seed: int = 0,
    model_path: Path | None = None,
) -> None:
    """
    Print the requested rows of the results table, evaluated on the validation split.

    The subset options select a smaller subset for fast iteration; with none of them it
    is the full evaluation. Subset scores are only comparable to other scores on the
    same subset, not to the full-split table.

    :param methods: Which rows to evaluate, in table order; all of them if omitted. Only
        the requested rows' indexes are built.
    :param train_cases: Sample this many train cases (with citations) for the constant
        baseline; all of them if omitted.
    :param val_queries: Sample this many validation queries (with citations); all of
        them if omitted.
    :param restrict_corpus: Retrieve from a reduced corpus: only the rulings that the
        (sampled) train and validation cases cite. Easier than the full corpus, since
        every ruling in it is cited by someone.
    :param seed: The sampling seed.
    :param model_path: A fine-tuned model's directory to use for the dense row instead
        of the pretrained model (see `scripts.train`).
    """
    selected = set(methods) if methods else set(Method)

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

    results: dict[Method, Scores] = {}
    for method in Method:
        if method not in selected:
            continue
        retrieve_fn = build_retrieve_fn(
            method=method,
            train_cases=train,
            ruling_ids=ruling_ids,
            model_path=model_path,
            k=100,
        )
        cases: Iterable[Case] = (
            val_cases
            if method == Method.CONSTANT
            else tqdm(val_cases, desc=f"evaluating {_ROW_LABELS[method]}")
        )
        results[method] = _evaluate(retrieve_fn=retrieve_fn, cases=cases)

    print(f"queries evaluated: {len(val_cases)}")
    print()
    header = f"{'method':11}" + "".join(
        f"{column.header:>11}" for column in _METRIC_COLUMNS
    )
    print(header)
    for method in Method:
        if method in results:
            label = _ROW_LABELS[method]
            if method == Method.DENSE and model_path is not None:
                label = "dense-ft"
            _print_row(label=label, scores=results[method])


if __name__ == "__main__":
    cyclopts.run(evaluate)
