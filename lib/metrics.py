"""
Retrieval metrics: Recall@k, nDCG@k, MRR, and a most-frequent-label baseline.

All three metrics use binary relevance - an item is either in `gold` or not, no
graded relevance - since that's what `cited_rulings`/`laws` give us.
"""

import math
from collections import Counter
from collections.abc import Iterable, Sequence


def top_k_recall(*, retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """
    The fraction of `gold` items present in the top `k` of `retrieved`.

    :param retrieved: Retrieved item ids, best first.
    :param gold: The relevant item ids for this query. Must be non-empty - a query
        with no relevant items has no meaningful recall.
    :param k: How many of the top-ranked `retrieved` items to consider.
    :returns: A value in [0, 1].
    """
    if not gold:
        raise ValueError("gold must be non-empty - recall is undefined for it")
    n_hits = len(set(retrieved[:k]) & gold)
    return n_hits / len(gold)


def _dcg(*, relevances: Iterable[int]) -> float:
    """
    Discounted Cumulative Gain: each relevance is discounted by log2(rank + 1),
    rewarding relevant items ranked earlier over ones ranked later.

    :param relevances: Relevance values in rank order (first = top-ranked).
    :returns: The (unnormalized) DCG.
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _binary_relevance(*, retrieved: Sequence[str], gold: set[str]) -> list[int]:
    """
    :param retrieved: Retrieved item ids, best first.
    :param gold: The relevant item ids for this query.
    :returns: 1 for each retrieved item that's in `gold`, 0 otherwise, in order.
    """
    return [1 if item in gold else 0 for item in retrieved]


def _idcg(*, gold: set[str], k: int) -> float:
    """
    The best possible DCG@k for a query with `gold` relevant items: all of them ranked
    first, up to `k` of them.

    :param gold: The relevant item ids for this query.
    :param k: How many of the top-ranked positions to consider.
    :returns: The ideal DCG@k.
    """
    return _dcg(relevances=[1] * min(len(gold), k))


def top_k_ndcg(*, retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at `k`, with binary relevance.

    :param retrieved: Retrieved item ids, best first.
    :param gold: The relevant item ids for this query. Must be non-empty.
    :param k: How many of the top-ranked `retrieved` items to consider.
    :returns: A value in [0, 1].
    """
    if not gold:
        raise ValueError("gold must be non-empty - nDCG is undefined for it")

    dcg = _dcg(relevances=_binary_relevance(retrieved=retrieved[:k], gold=gold))
    idcg = _idcg(gold=gold, k=k)
    return dcg / idcg if idcg else 0.0


def mrr(*, retrieved: Sequence[str], gold: set[str]) -> float:
    """
    Mean (here: single-query) Reciprocal Rank: 1 / (rank of the first relevant item),

    0 if none of `retrieved` is relevant. Averaging this over queries is the
    caller's job - this computes it for one query.

    :param retrieved: Retrieved item ids, best first.
    :param gold: The relevant item ids for this query. Must be non-empty.
    :returns: A value in [0, 1].
    """
    if not gold:
        raise ValueError("gold must be non-empty - MRR is undefined for it")
    for rank in range(1, len(retrieved) + 1):
        if retrieved[rank - 1] in gold:
            return 1 / rank
    return 0.0


def most_common_baseline(*, gold_sets: Iterable[set[str]], k: int) -> list[str]:
    """
    The `k` item ids that appear most often across `gold_sets` - a query-blind,
    constant prediction to compare real retrieval results against (this alone gets Recall@10 = 0.708 on `laws`).

    :param gold_sets: One collection of relevant item ids per query, e.g. every
        training-split `case.cited_rulings`.
    :param k: How many of the most common ids to return.
    :returns: The `k` most common ids, most common first.
    """
    counts: Counter[str] = Counter()
    for gold in gold_sets:
        counts.update(gold)
    return [item for item, _ in counts.most_common(k)]
