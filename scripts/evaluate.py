"""Evaluate the BM25 and zero-shot dense baselines against the validation split's
cited_rulings target, alongside the most-common-rulings constant baseline for
comparison."""

import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cyclopts
from tqdm import tqdm

from lib.dataset import Case, Split, iter_cases
from lib.metrics import mrr, top_k_ndcg, top_k_recall
from lib.retrieval_methods import Method, build_retrieve_fn
from lib.rulings import list_ruling_ids
from lib.sampling import sample_items
from paths import EVALUATIONS_DIR

_ROW_LABELS = {
    Method.CONSTANT: "constant",
    Method.BM25: "BM25",
    Method.BM25_CHUNKED: "BM25-chunk",
    Method.IDF_OVERLAP: "IDF-overlap",
    Method.DENSE: "dense",
    Method.HYBRID: "hybrid",
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
    *,
    retrieve_fn: Callable[[Case], list[str]],
    cases: Iterable[Case],
    progress: tqdm | None = None,
) -> Scores:
    """
    Compute mean Recall@10/50/100, nDCG@10, and MRR over `cases`.

    :param retrieve_fn: Given a case, returns up to 100 ranked ruling decision_ids.
    :param cases: Cases to evaluate over - each must have non-empty cited_rulings.
    :param progress: The progress bar `cases` is wrapped in, if any; its postfix shows
        the running means so far.
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
        if progress is not None:
            progress.set_postfix(
                R10=f"{statistics.mean(recall_10):.3f}",
                MRR=f"{statistics.mean(mrrs):.3f}",
            )
    return Scores(
        recall_at_10=statistics.mean(recall_10),
        recall_at_50=statistics.mean(recall_50),
        recall_at_100=statistics.mean(recall_100),
        ndcg_at_10=statistics.mean(ndcg_10),
        mrr=statistics.mean(mrrs),
    )


def _emit(*, line: str, log_path: Path) -> None:
    """
    Print a line and append it to the log file, so it survives a killed run.

    :param line: The line to write.
    :param log_path: The log file to append to.
    """
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(line + "\n")


def _header_line() -> str:
    return f"{'method':11}" + "".join(
        f"{column.header:>11}" for column in _METRIC_COLUMNS
    )


def _row_line(*, label: str, scores: Scores) -> str:
    """
    :param label: The method name for the row's first column.
    :param scores: The row's metric values.
    :returns: The row's text.
    """
    values = "".join(
        f"{getattr(scores, column.attribute):>11.3f}" for column in _METRIC_COLUMNS
    )
    return f"{label:11}{values}"


def evaluate(
    *,
    methods: list[Method] | None = None,
    train_cases: int | None = None,
    val_queries: int | None = None,
    restrict_corpus: bool = False,
    seed: int = 0,
    model_path: Path | None = None,
    log_path: Path | None = None,
) -> None:
    """
    Print the requested rows of the results table, evaluated on the validation split.

    Each row is printed as soon as its method is done and appended to a log file, and
    the progress bar shows the running Recall@10 and MRR, so a long run can be read
    before it finishes. The subset options select a smaller subset for fast iteration;
    with none of them it is the full evaluation. Subset scores are only comparable to
    other scores on the same subset, not to the full-split table.

    :param methods: Which rows to evaluate, in table order; all but the hybrid if
        omitted (it combines two of the others, and is meant for a fine-tuned model, so
        it is asked for explicitly). Only the requested rows' indexes are built.
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
    :param log_path: The file the settings and result rows are appended to; a new
        timestamped file under data/evaluations/ if omitted.
    """
    selected = set(methods) if methods else set(Method) - {Method.HYBRID}

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

    if log_path is None:
        log_path = EVALUATIONS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {log_path}", flush=True)
    settings = (
        f"train_cases={train_cases} val_queries={val_queries} "
        f"restrict_corpus={restrict_corpus} seed={seed} model_path={model_path}"
    )
    _emit(line=settings, log_path=log_path)
    _emit(line=f"queries evaluated: {len(val_cases)}\n", log_path=log_path)
    _emit(line=_header_line(), log_path=log_path)
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
        label = _ROW_LABELS[method]
        if method in {Method.DENSE, Method.HYBRID} and model_path is not None:
            label = f"{label}-ft"
        if method == Method.CONSTANT:
            scores = _evaluate(retrieve_fn=retrieve_fn, cases=val_cases)
        else:
            progress = tqdm(val_cases, desc=f"evaluating {label}")
            scores = _evaluate(
                retrieve_fn=retrieve_fn, cases=progress, progress=progress
            )
        _emit(line=_row_line(label=label, scores=scores), log_path=log_path)


if __name__ == "__main__":
    cyclopts.run(evaluate)
