"""
Building a retrieval function for a chosen method.

Shared by scripts.evaluate and failure_analysis.inspect_failures, so both run the exact
same methods the exact same way.
"""

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from lib.chunking import scores_by_document
from lib.dataset import Case
from lib.dense.retrieval import build_dense_index
from lib.dense.retrieval import retrieve as dense_retrieve
from lib.dense.retrieval import score_chunks as dense_score_chunks
from lib.lexical.bm25 import build_ruling_index
from lib.lexical.bm25 import retrieve as bm25_retrieve
from lib.lexical.bm25_chunked import build_chunk_index
from lib.lexical.bm25_chunked import retrieve as bm25_chunked_retrieve
from lib.lexical.bm25_chunked import score_chunks
from lib.lexical.idf_overlap import build_overlap_index
from lib.lexical.idf_overlap import retrieve as idf_overlap_retrieve
from lib.metrics import most_common_baseline
from lib.query import build_query


class Method(StrEnum):
    CONSTANT = "constant"
    BM25 = "bm25"
    BM25_CHUNKED = "bm25-chunked"
    IDF_OVERLAP = "idf-overlap"
    DENSE = "dense"


def build_retrieve_fn(
    *,
    method: Method,
    train_cases: list[Case],
    ruling_ids: frozenset[str] | None,
    model_path: Path | None,
    k: int,
) -> Callable[[Case], list[str]]:
    """
    Build `method`'s one-time index, then a function from a case to its top-k retrieved
    ruling decision_ids.

    :param method: Which retrieval method.
    :param train_cases: Training cases with citations, for the constant baseline's most-
        common ranking. Ignored by the other methods.
    :param ruling_ids: Retrieve from only these rulings; the full corpus if None.
    :param model_path: A fine-tuned model's directory for the dense method; the
        pretrained model if None. Ignored by the other methods.
    :param k: How many results the returned function returns per case.
    :returns: A function from a `Case` to its top-k ranked ruling decision_ids.
    """
    if method == Method.CONSTANT:
        baseline_top_k = most_common_baseline(
            gold_sets=(set(c.cited_rulings) for c in train_cases), k=k
        )
        return lambda _case: baseline_top_k
    if method == Method.BM25:
        ruling_index = build_ruling_index(ruling_ids=ruling_ids)
        return lambda case: bm25_retrieve(
            ruling_index, build_query(case=case), case.language, k=k
        )
    if method == Method.BM25_CHUNKED:
        chunk_index = build_chunk_index(ruling_ids=ruling_ids)
        return lambda case: bm25_chunked_retrieve(
            chunk_index=chunk_index,
            query=build_query(case=case),
            language=case.language,
            k=k,
        )
    if method == Method.IDF_OVERLAP:
        overlap_index = build_overlap_index(ruling_ids=ruling_ids)
        return lambda case: idf_overlap_retrieve(
            overlap_index=overlap_index,
            query=build_query(case=case),
            language=case.language,
            k=k,
        )
    dense_index = build_dense_index(ruling_ids=ruling_ids, model_path=model_path)
    return lambda case: dense_retrieve(
        index=dense_index, query=build_query(case=case), k=k, model_path=model_path
    )


def build_chunk_score_fn(
    *, method: Method, ruling_ids: frozenset[str] | None, model_path: Path | None
) -> Callable[[Case, list[str]], dict[str, list[float]]] | None:
    """
    Build `method`'s function from a case and some rulings to the score of each of those
    rulings' chunks - the numbers `build_retrieve_fn`'s ranking takes the max of.

    :param method: Which retrieval method.
    :param ruling_ids: The corpus the method retrieves from; the full corpus if None.
    :param model_path: A fine-tuned model's directory for the dense method; the
        pretrained model if None. Ignored by the other methods.
    :returns: The function, or None for methods that don't score ruling chunks
        (constant and whole-ruling BM25). Given a case and ruling decision_ids, it
        returns each ruling's chunk scores, in chunk order.
    """
    if method == Method.BM25_CHUNKED:
        chunk_index = build_chunk_index(ruling_ids=ruling_ids)

        def bm25_chunk_scores(
            case: Case, wanted_ids: list[str]
        ) -> dict[str, list[float]]:
            return scores_by_document(
                chunk_scores=score_chunks(
                    chunk_index=chunk_index,
                    query=build_query(case=case),
                    language=case.language,
                ),
                chunk_starts=chunk_index.chunk_starts,
                document_ids=chunk_index.decision_ids,
                wanted_ids=wanted_ids,
            )

        return bm25_chunk_scores
    if method == Method.DENSE:
        dense_index = build_dense_index(ruling_ids=ruling_ids, model_path=model_path)

        def dense_chunk_scores(
            case: Case, wanted_ids: list[str]
        ) -> dict[str, list[float]]:
            return scores_by_document(
                chunk_scores=dense_score_chunks(
                    index=dense_index,
                    query=build_query(case=case),
                    model_path=model_path,
                ),
                chunk_starts=dense_index.chunk_starts,
                document_ids=dense_index.decision_ids,
                wanted_ids=wanted_ids,
            )

        return dense_chunk_scores
    return None
