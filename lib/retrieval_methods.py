"""
Building a retrieval function for a chosen method.

Used by scripts.evaluate, so every method is built the same way.
"""

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from lib.dataset import Case
from lib.dense.retrieval import build_dense_index
from lib.dense.retrieval import retrieve as dense_retrieve
from lib.fusion import reciprocal_rank_fusion
from lib.lexical.bm25 import build_ruling_index
from lib.lexical.bm25 import retrieve as bm25_retrieve
from lib.lexical.bm25_chunked import build_chunk_index
from lib.lexical.bm25_chunked import retrieve as bm25_chunked_retrieve
from lib.lexical.idf_overlap import build_overlap_index
from lib.lexical.idf_overlap import retrieve as idf_overlap_retrieve
from lib.metrics import most_common_baseline
from lib.query import build_query


# How deep each method's ranking is for the hybrid: deeper than the final `k`, so that
# an item one method ranks low can still be lifted by the other.
_HYBRID_POOL = 200


class Method(StrEnum):
    CONSTANT = "constant"
    BM25 = "bm25"
    BM25_CHUNKED = "bm25-chunked"
    IDF_OVERLAP = "idf-overlap"
    DENSE = "dense"
    HYBRID = "hybrid"


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
    :param model_path: A fine-tuned model's directory for the dense and hybrid
        methods; the pretrained model if None. Ignored by the other methods.
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
    if method == Method.HYBRID:
        pool = max(k, _HYBRID_POOL)
        lexical_fn = build_retrieve_fn(
            method=Method.BM25_CHUNKED,
            train_cases=train_cases,
            ruling_ids=ruling_ids,
            model_path=model_path,
            k=pool,
        )
        dense_fn = build_retrieve_fn(
            method=Method.DENSE,
            train_cases=train_cases,
            ruling_ids=ruling_ids,
            model_path=model_path,
            k=pool,
        )
        return lambda case: reciprocal_rank_fusion(
            rankings=[lexical_fn(case), dense_fn(case)], k=k
        )
    dense_index = build_dense_index(ruling_ids=ruling_ids, model_path=model_path)
    return lambda case: dense_retrieve(
        index=dense_index, query=build_query(case=case), k=k, model_path=model_path
    )
