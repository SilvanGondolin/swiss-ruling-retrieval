import numpy as np
import pytest

from lib.chunking import TokenSpan, chunk_starts, chunk_text, scores_by_document

# "Das ist ein Test." tokenized by multilingual-e5-small, without special tokens
_TEXT = "Das ist ein Test."
_OFFSETS = [
    TokenSpan(start=0, end=3),
    TokenSpan(start=4, end=7),
    TokenSpan(start=8, end=11),
    TokenSpan(start=12, end=16),
    TokenSpan(start=16, end=17),
]


def test_chunk_text_splits_into_windows_of_window_size_tokens():
    assert chunk_text(text=_TEXT, token_offsets=_OFFSETS, window_size=2) == [
        "Das ist",
        "ein Test",
        ".",
    ]


def test_chunk_text_returns_one_chunk_when_text_fits_the_window():
    assert chunk_text(text=_TEXT, token_offsets=_OFFSETS, window_size=5) == [_TEXT]
    assert chunk_text(text=_TEXT, token_offsets=_OFFSETS, window_size=99) == [_TEXT]


def test_chunk_text_keeps_a_text_without_tokens_as_one_chunk():
    assert chunk_text(text="", token_offsets=[], window_size=4) == [""]


def test_chunk_text_rejects_non_positive_window_size():
    with pytest.raises(ValueError):
        chunk_text(text=_TEXT, token_offsets=_OFFSETS, window_size=0)


def test_chunk_starts_are_cumulative_chunk_counts():
    assert chunk_starts(chunk_counts=[2, 1, 3]).tolist() == [0, 2, 3]


def test_chunk_starts_of_no_documents_is_empty():
    assert len(chunk_starts(chunk_counts=[])) == 0


def test_chunk_starts_rejects_a_document_without_chunks():
    with pytest.raises(ValueError):
        chunk_starts(chunk_counts=[2, 0])


def test_chunk_starts_is_int_array():
    assert chunk_starts(chunk_counts=[1, 1]).dtype == np.int64


def test_scores_by_document_slices_each_documents_chunks():
    scores = scores_by_document(
        chunk_scores=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        chunk_starts=np.array([0, 2, 3]),
        document_ids=["a", "b", "c"],
        wanted_ids=["c", "a", "b"],
    )
    assert scores == {"a": [0.1, 0.2], "b": [0.3], "c": [0.4, 0.5, 0.6]}


def test_scores_by_document_returns_only_the_wanted_documents():
    scores = scores_by_document(
        chunk_scores=np.array([0.1, 0.2]),
        chunk_starts=np.array([0, 1]),
        document_ids=["a", "b"],
        wanted_ids=["b"],
    )
    assert scores == {"b": [0.2]}
