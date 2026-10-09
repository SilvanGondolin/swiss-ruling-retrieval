import bm25s
import numpy as np

from lib.language import Language
from lib.lexical.bm25 import document_frequencies, tokenize
from lib.lexical.idf_overlap import OverlapIndex, overlap_scores, retrieve


def _index(*, documents: list[list[str]]) -> bm25s.BM25:
    index = bm25s.BM25()
    index.index(documents, show_progress=False)
    return index


def test_overlap_scores_sum_the_weights_of_the_stems_each_document_contains():
    index = _index(documents=[["a", "b"], ["b", "c"], ["c"]])
    scores = overlap_scores(index=index, stem_weights={"a": 5.0, "b": 2.0})
    np.testing.assert_allclose(scores, [7.0, 2.0, 0.0])


def test_overlap_scores_count_a_stem_once_however_often_it_occurs():
    index = _index(documents=[["a", "a", "a"], ["a", "b"]])
    scores = overlap_scores(index=index, stem_weights={"a": 1.0})
    np.testing.assert_allclose(scores, [1.0, 1.0])


def test_overlap_scores_ignore_stems_absent_from_the_index():
    index = _index(documents=[["a"], ["b"]])
    scores = overlap_scores(index=index, stem_weights={"zzz": 9.0, "a": 1.0})
    np.testing.assert_allclose(scores, [1.0, 0.0])


def _overlap_index(*, texts: dict[str, str]) -> OverlapIndex:
    index = _index(documents=[tokenize(text, Language.DE) for text in texts.values()])
    return OverlapIndex(
        index=index,
        decision_ids=list(texts),
        frequencies=document_frequencies(
            index=index, document_languages=[Language.DE] * len(texts)
        ),
    )


def test_retrieve_ranks_rulings_by_shared_stems_best_first():
    overlap_index = _overlap_index(
        texts={
            "none": "Mietvertrag Kündigung",
            "one": "Sicherheitshaft Kündigung",
            "two": "Sicherheitshaft Verwahrung Rückfallgefahr",
        }
    )
    assert retrieve(
        overlap_index=overlap_index,
        query="Sicherheitshaft Verwahrung",
        language=Language.DE,
        k=3,
    ) == ["two", "one", "none"]


def test_retrieve_weights_a_rare_stem_above_a_common_one():
    overlap_index = _overlap_index(
        texts={
            "common": "Beschwerde",
            "rare": "Verwahrung",
            "filler1": "Beschwerde",
            "filler2": "Beschwerde",
        }
    )
    # "Beschwerde" is in 3 of 4 rulings, "Verwahrung" in 1: one rare shared stem beats
    # one common shared stem, though "common" comes first in the index
    assert retrieve(
        overlap_index=overlap_index,
        query="Beschwerde Verwahrung",
        language=Language.DE,
        k=1,
    ) == ["rare"]


def test_retrieve_respects_k_and_keeps_index_order_on_ties():
    overlap_index = _overlap_index(texts={"x": "Haft", "y": "Haft", "z": "Haft"})
    assert retrieve(
        overlap_index=overlap_index, query="Haft", language=Language.DE, k=2
    ) == ["x", "y"]
