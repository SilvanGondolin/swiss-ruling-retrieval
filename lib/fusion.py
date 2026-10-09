"""Combining the rankings of several retrieval methods into one."""

from collections import defaultdict
from collections.abc import Sequence

# The constant of reciprocal rank fusion (Cormack et al., 2009); 60 is the value from
# that paper and is used as is, not tuned on the validation split.
RRF_CONSTANT = 60


def reciprocal_rank_fusion(
    *, rankings: Sequence[Sequence[str]], k: int, constant: int = RRF_CONSTANT
) -> list[str]:
    """
    Fuse rankings by reciprocal rank: an item's score is the sum over the rankings of
    1 / (constant + rank), with rank 1 for the best item, and an item missing from a
    ranking gets nothing from it.

    :param rankings: One ranking per method, best first, each without duplicates.
    :param k: How many fused items to return.
    :param constant: Larger values flatten the difference between top ranks.
    :returns: Up to `k` item ids, best first; ties keep the order of first appearance.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (constant + rank)
    return sorted(scores, key=lambda item: -scores[item])[:k]
