"""
Write a method's worst retrieval failures as a static HTML report - the browsable,
filterable counterpart to the console tool `failure_analysis.inspect_failures`.
"""

from pathlib import Path

import cyclopts

from failure_analysis.html_report import render_report
from failure_analysis.worst_cases import collect_worst_cases
from lib.lexical.bm25 import (
    DocumentFrequencies,
    build_ruling_index,
    document_frequencies,
)
from lib.retrieval_methods import Method
from lib.rulings import iter_rulings
from paths import DATA_DIR


def _corpus_frequencies() -> DocumentFrequencies:
    """
    :returns: Document frequencies over the whole ruling corpus, read off the full BM25
        index (built and cached on first use, which takes a while).
    """
    ruling_index = build_ruling_index()
    languages_by_id = {ruling.decision_id: ruling.language for ruling in iter_rulings()}
    return document_frequencies(
        index=ruling_index.index,
        document_languages=[
            languages_by_id[decision_id] for decision_id in ruling_index.decision_ids
        ],
    )


def report(
    *,
    method: Method,
    n: int | None = None,
    k: int = 10,
    output: Path | None = None,
    train_cases: int | None = None,
    val_queries: int | None = None,
    restrict_corpus: bool = False,
    seed: int = 0,
    model_path: Path | None = None,
) -> None:
    """
    Write the validation cases where `method` has the lowest Recall@k to a static HTML
    report: a table of cases, each opening a side-by-side view of the query and one of
    its rulings, for reading why they failed. The page embeds every shown query and
    ruling text, so use `val_queries` to keep it a manageable size (500 cases is about
    40 MB). Shared words are rated by rarity over the whole corpus, which needs the
    full-corpus BM25 index, whatever method is inspected.

    :param method: Which retrieval method to inspect.
    :param n: How many of the worst cases to include; all evaluated cases if omitted.
    :param k: How many results `method` retrieves per case.
    :param output: Where to write the report; data/reports/<method>_failures.html if
        omitted.
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
    html = render_report(
        worst_cases=worst_cases,
        method_label=method.value,
        frequencies=_corpus_frequencies(),
    )

    out_path = output or DATA_DIR / "reports" / f"{method.value}_failures.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    cyclopts.run(report)
