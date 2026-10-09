from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lib.language import Language
from lib.lexical.bm25_chunked import (
    ChunkIndex,
    build_chunk_index,
    retrieve,
    score_chunks,
)
from lib.ruling_chunks import RulingChunks


def _ruling_chunks() -> RulingChunks:
    # "mietrecht" (rental law) has two chunks, of which only the second is about
    # eviction; "strafrecht" (criminal law) has one chunk; "droit" is a French ruling
    # with two chunks
    return RulingChunks(
        decision_ids=["mietrecht", "strafrecht", "droit"],
        languages=[Language.DE, Language.DE, Language.FR],
        chunk_starts=np.array([0, 2, 3]),
        texts=[
            "Verfahren Sachverhalt Beschwerdeführer Vorinstanz",
            "Mietvertrag Kündigung Mieter Vermieter Mietzins",
            "Strafverfahren Freiheitsstrafe Verurteilung Täter Busse",
            "responsabilité contractuelle bailleur locataire dommages",
            "procédure recours tribunal fédéral",
        ],
    )


@pytest.fixture
def chunk_index(tmp_path, monkeypatch):
    monkeypatch.setattr("lib.lexical.bm25_chunked.INDEX_DIR", tmp_path)
    with patch(
        "lib.lexical.bm25_chunked.load_ruling_chunks", return_value=_ruling_chunks()
    ):
        return build_chunk_index()


def test_build_chunk_index_keeps_the_ruling_chunk_layout(chunk_index):
    assert chunk_index.decision_ids == ["mietrecht", "strafrecht", "droit"]
    assert chunk_index.chunk_starts.tolist() == [0, 2, 3]


def test_build_chunk_index_caches_and_reloads(tmp_path, monkeypatch, chunk_index):
    assert (tmp_path / "bm25_chunk_index").is_dir()
    monkeypatch.setattr("lib.lexical.bm25_chunked.INDEX_DIR", tmp_path)

    # the second call must come from the cache, not from the corpus
    with patch(
        "lib.lexical.bm25_chunked.load_ruling_chunks",
        side_effect=AssertionError("not cached"),
    ):
        loaded = build_chunk_index()

    assert loaded.decision_ids == chunk_index.decision_ids
    assert loaded.chunk_starts.tolist() == chunk_index.chunk_starts.tolist()
    query = "Kündigung Mieter"
    np.testing.assert_allclose(
        score_chunks(chunk_index=loaded, query=query, language=Language.DE),
        score_chunks(chunk_index=chunk_index, query=query, language=Language.DE),
    )


def test_build_chunk_index_of_a_subset_uses_its_own_cache_directory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("lib.lexical.bm25_chunked.INDEX_DIR", tmp_path)
    subset = frozenset({"mietrecht", "strafrecht"})
    with patch(
        "lib.lexical.bm25_chunked.load_ruling_chunks", return_value=_ruling_chunks()
    ) as load:
        build_chunk_index(ruling_ids=subset)

    load.assert_called_once_with(ruling_ids=subset)
    [cache_dir] = tmp_path.iterdir()
    assert cache_dir.name.startswith("bm25_chunk_index_n2_")


def test_score_chunks_scores_every_chunk_and_ranks_the_matching_one_highest(
    chunk_index,
):
    scores = score_chunks(
        chunk_index=chunk_index, query="Kündigung Mieter", language=Language.DE
    )
    assert scores.shape == (5,)
    assert int(np.argmax(scores)) == 1
    assert scores[1] > 0


def test_score_chunks_tokenizes_each_chunk_with_its_rulings_language(chunk_index):
    # the French stemmer turns "responsabilité contractuelle" into "respons" and
    # "contractuel", the German one into "responsabilité" and "contractuell": the
    # French chunk only matches if it was indexed with the French stemmer
    scores = score_chunks(
        chunk_index=chunk_index,
        query="responsabilité contractuelle",
        language=Language.FR,
    )
    assert int(np.argmax(scores)) == 3
    assert scores[3] > 0


def test_score_chunks_of_a_query_without_content_words_is_all_zeros(chunk_index):
    # every token is a German stopword, so nothing is left to match
    scores = score_chunks(
        chunk_index=chunk_index, query="der die und", language=Language.DE
    )
    assert scores.tolist() == [0.0] * 5


def test_score_chunks_of_a_query_unknown_to_the_index_is_all_zeros(chunk_index):
    scores = score_chunks(
        chunk_index=chunk_index, query="Quantenchromodynamik", language=Language.DE
    )
    assert scores.tolist() == [0.0] * 5


def test_retrieve_scores_a_ruling_by_its_best_chunk():
    # chunk scores per ruling: A [4, 0], B [3, 3], C [0, 5]. By best chunk the order is
    # C (5), A (4), B (3). The sum of chunks would give B, C, A; the first chunk A, B, C;
    # the last chunk C, B, A
    chunk_index = ChunkIndex(
        index=MagicMock(),
        chunk_starts=np.array([0, 2, 4]),
        decision_ids=["A", "B", "C"],
    )
    chunk_scores = np.array([4.0, 0.0, 3.0, 3.0, 0.0, 5.0])
    with patch("lib.lexical.bm25_chunked.score_chunks", return_value=chunk_scores):
        results = retrieve(
            chunk_index=chunk_index, query="ignored", language=Language.DE, k=3
        )
    assert results == ["C", "A", "B"]


def test_retrieve_ranks_the_matching_ruling_first_for_each_topic(chunk_index):
    results = retrieve(
        chunk_index=chunk_index,
        query="Freiheitsstrafe und Verurteilung des Täters",
        language=Language.DE,
        k=3,
    )
    assert results[0] == "strafrecht"


def test_retrieve_respects_k(chunk_index):
    results = retrieve(
        chunk_index=chunk_index, query="Kündigung Mieter", language=Language.DE, k=2
    )
    assert len(results) == 2
    assert results[0] == "mietrecht"
