"""
Batch sampling for contrastive training with in-batch negatives.

With in-batch negatives, every other pair's ruling chunk is a "negative" for a query.
That is wrong - a false negative - when the other pair comes from the same case (its
other cited rulings are relevant to the same query) or cites the same ruling (the
same ruling is relevant to both). So batches must never repeat a case or a ruling.
"""

import random
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from sentence_transformers.base.sampler import DefaultBatchSampler


def distinct_batches(
    *,
    case_ids: Sequence[str],
    ruling_ids: Sequence[str],
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    """
    Partition pair indices (in a shuffled order) into full batches in which no case id
    and no ruling id occurs twice.

    A pair that clashes with its batch is held back and offered again to the next
    batch. Pairs still unplaced at the end - and a final incomplete batch - are
    dropped, so a ruling cited by more pairs than there are batches never has all its
    pairs used in one pass. Callers shuffle differently each epoch (via `seed`), so
    different pairs are dropped each time.

    :param case_ids: The citing case of each pair.
    :param ruling_ids: The cited ruling of each pair.
    :param batch_size: The size of every batch.
    :param seed: The shuffling seed.
    :returns: Batches of pair indices, each of exactly `batch_size`.
    """
    order = list(range(len(case_ids)))
    random.Random(seed).shuffle(order)
    queue = deque(order)
    batches: list[list[int]] = []
    current: list[int] = []
    used_cases: set[str] = set()
    used_rulings: set[str] = set()
    held_back: list[int] = []
    while queue:
        i = queue.popleft()
        if case_ids[i] in used_cases or ruling_ids[i] in used_rulings:
            held_back.append(i)
            continue
        current.append(i)
        used_cases.add(case_ids[i])
        used_rulings.add(ruling_ids[i])
        if len(current) == batch_size:
            batches.append(current)
            current, used_cases, used_rulings = [], set(), set()
            queue = deque([*held_back, *queue])
            held_back = []
    return batches


class DistinctBatchSampler(DefaultBatchSampler):
    """Yields `distinct_batches` over the training dataset, reshuffled each epoch."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        case_ids: Sequence[str],
        ruling_ids: Sequence[str],
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None,
        generator: torch.Generator | None,
        seed: int,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        self._case_ids = case_ids
        self._ruling_ids = ruling_ids
        self._n_batches: int | None = None

    def _batches(self, *, seed: int) -> list[list[int]]:
        return distinct_batches(
            case_ids=self._case_ids,
            ruling_ids=self._ruling_ids,
            batch_size=self.batch_size,
            seed=seed,
        )

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches(seed=self.seed + getattr(self, "epoch", 0))

    def __len__(self) -> int:
        # The batch count varies slightly with the shuffle; the first epoch's count is
        # what the training schedule is sized from.
        if self._n_batches is None:
            self._n_batches = len(self._batches(seed=self.seed))
        return self._n_batches


@dataclass(frozen=True)
class DistinctSamplerFactory:
    """
    The trainer's `batch_sampler`: builds a `DistinctBatchSampler` for the dataset the
    trainer hands it. A class rather than a closure, as the trainer pickles its
    arguments into every checkpoint.

    :ivar case_ids: The citing case of each training row, in dataset order.
    :ivar ruling_ids: The cited ruling of each training row, in dataset order.
    """

    case_ids: Sequence[str]
    ruling_ids: Sequence[str]

    # **kwargs: sentence-transformers' batch_sampler factory protocol passes through
    # whatever DefaultBatchSampler's other constructor arguments it builds internally -
    # forwarded verbatim to DistinctBatchSampler.__init__, which is itself fully typed.
    def __call__(self, dataset: Dataset, **kwargs: Any) -> DistinctBatchSampler:
        return DistinctBatchSampler(
            dataset, case_ids=self.case_ids, ruling_ids=self.ruling_ids, **kwargs
        )
