"""Deterministic random subsets, for fast iteration on a fraction of the data."""

import random
from collections.abc import Sequence
from typing import TypeVar

_T = TypeVar("_T")


def sample_items(*, items: Sequence[_T], n: int | None, seed: int) -> list[_T]:
    """
    :param items: The population, in a deterministic order.
    :param n: How many to sample; everything if None or if the population is smaller.
    :param seed: The sampling seed.
    :returns: A reproducible random sample of `items`, in original order.
    """
    if n is None or n >= len(items):
        return list(items)
    chosen = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in chosen]
