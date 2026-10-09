"""Naming the on-disk cache of an index built over all or part of the ruling corpus."""

import hashlib


def cache_name(*, base: str, ruling_ids: frozenset[str] | None) -> str:
    """
    Derive a cache directory name from the corpus an index is built over, so that
    indexes over different rulings can never be mistaken for each other.

    :param base: The index kind's own name, e.g. "dense_index".
    :param ruling_ids: The rulings the index covers; None for the full corpus.
    :returns:`base` for the full corpus, otherwise `base` plus the number of rulings and
        a short hash of exactly which ones, e.g. "dense_index_n4684_3fa9c1d2e0".
    """
    if ruling_ids is None:
        return base
    digest = hashlib.sha1("\n".join(sorted(ruling_ids)).encode()).hexdigest()[:10]
    return f"{base}_n{len(ruling_ids)}_{digest}"
