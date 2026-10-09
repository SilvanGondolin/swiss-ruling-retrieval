"""
Word-overlap baseline: a ruling scores as the summed rarity of the word stems it shares.

with the query. Each stem counts once, however often it occurs, and a ruling's length
doesn't matter - BM25 without term-frequency saturation or length normalization, so
the baseline for "how far do rare shared words alone get you".

Reads its statistics off the BM25 ruling index (see `lib.lexical.bm25`): that index stores, per
stem, the rulings containing it.
"""

from dataclasses import dataclass

import bm25s
import numpy as np

from lib.language import Language
from lib.lexical.bm25 import (
    DocumentFrequencies,
    build_ruling_index,
    document_frequencies,
    tokenize,
)
from lib.rulings import iter_rulings


@dataclass(frozen=True)
class OverlapIndex:
    """
    :ivar index: The BM25 index of the rulings, used for which rulings contain a stem.
    :ivar decision_ids: The ruling for each indexed document.
    :ivar frequencies: How rare each stem is among the rulings.
    """

    index: bm25s.BM25
    decision_ids: list[str]
    frequencies: DocumentFrequencies


def build_overlap_index(*, ruling_ids: frozenset[str] | None = None) -> OverlapIndex:
    """
    :param ruling_ids: Only index these rulings (a reduced corpus, for fast
        iteration); all of them if omitted.
    :returns: The index, built on the (cached) BM25 ruling index - see
        `lib.lexical.bm25.build_ruling_index`, which this builds first if it isn't cached yet.
    """
    ruling_index = build_ruling_index(ruling_ids=ruling_ids)
    languages_by_id = {
        ruling.decision_id: ruling.language
        for ruling in iter_rulings(ruling_ids=ruling_ids)
    }
    frequencies = document_frequencies(
        index=ruling_index.index,
        document_languages=[
            languages_by_id[decision_id] for decision_id in ruling_index.decision_ids
        ],
    )
    return OverlapIndex(
        index=ruling_index.index,
        decision_ids=ruling_index.decision_ids,
        frequencies=frequencies,
    )


def overlap_scores(*, index: bm25s.BM25, stem_weights: dict[str, float]) -> np.ndarray:
    """
    :param index: A BM25 index of rulings.
    :param stem_weights: The weight of each query stem.
    :returns: For every indexed ruling, the summed weight of the `stem_weights` stems
        it contains. Stems absent from the index are ignored.
    """
    column_starts = index.scores["indptr"]
    row_indices = index.scores["indices"]
    scores = np.zeros(int(index.scores["num_docs"]))
    for stem, weight in stem_weights.items():
        stem_id = index.vocab_dict.get(stem)
        if stem_id is None or stem_id + 1 >= len(column_starts):
            continue
        rows = row_indices[column_starts[stem_id] : column_starts[stem_id + 1]]
        scores[rows] += weight
    return scores


def retrieve(
    *, overlap_index: OverlapIndex, query: str, language: Language, k: int
) -> list[str]:
    """
    Retrieve the top-k ruling decision_ids for `query` by rarity-weighted stem overlap.

    :param overlap_index: An index built by `build_overlap_index`.
    :param query: The query text - already citation-masked, see `lib.query`.
    :param language: The query's language.
    :param k: How many results to return.
    :returns: Up to `k` decision_ids, best first; ties keep index order.
    """
    stem_weights = {
        stem: overlap_index.frequencies.idf(stem=stem, language=language)
        for stem in set(tokenize(query, language))
    }
    scores = overlap_scores(index=overlap_index.index, stem_weights=stem_weights)
    top_k_idx = np.argsort(-scores, kind="stable")[:k]
    return [overlap_index.decision_ids[i] for i in top_k_idx]
