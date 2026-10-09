from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from lib.dense.retrieval import DenseIndex, _embed_query, retrieve


def _index() -> DenseIndex:
    # ruling "a" has chunks 0-1, "b" has chunk 2, "c" has chunks 3-4.
    # Best-chunk score per ruling:
    #   query [1, 0]:  a = 1.0 (its 2nd chunk), c = 0.71 (its 2nd chunk), b = 0.6
    #   query [-1, 0]: c = 1.0 (its 1st chunk), a = 0.0 (its 1st chunk), b = -0.6
    chunk_embeddings = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.6, 0.8],
            [-1.0, 0.0],
            [0.7071, 0.7071],
        ]
    )
    return DenseIndex(
        chunk_embeddings=chunk_embeddings,
        chunk_starts=np.array([0, 2, 3]),
        decision_ids=["a", "b", "c"],
    )


def _retrieve_with_query_embedding(*, query_embedding: list[float], k: int):
    with patch(
        "lib.dense.retrieval._embed_query",
        return_value=np.array(query_embedding),
    ):
        return retrieve(index=_index(), query="ignored, embedding is patched", k=k)


def test_retrieve_ranks_by_best_matching_chunk_best_first():
    # mean-pooling the chunks would rank "b" (0.6) above "a" (mean 0.5), and scoring
    # by the first chunk only would put "b" (0.6) above "a" (0.0) too
    assert _retrieve_with_query_embedding(query_embedding=[1.0, 0.0], k=3) == [
        "a",
        "c",
        "b",
    ]


def test_retrieve_finds_the_best_chunk_wherever_it_is_in_the_ruling():
    # here the best chunk of "c" and "a" is their first one; scoring by the last
    # chunk only would rank "b" (-0.6) first, then "c" (-0.71), then "a" (-1.0)
    assert _retrieve_with_query_embedding(query_embedding=[-1.0, 0.0], k=3) == [
        "c",
        "a",
        "b",
    ]


def test_retrieve_respects_k():
    assert _retrieve_with_query_embedding(query_embedding=[1.0, 0.0], k=1) == ["a"]


def _embed_query_with_chunk_embeddings(*, chunk_embeddings: list[list[float]]):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(chunk_embeddings)
    chunks = [f"chunk {i}" for i in range(len(chunk_embeddings))]
    with (
        patch("lib.dense.retrieval.model", return_value=mock_model),
        patch("lib.dense.retrieval.chunk_for_model", return_value=chunks),
    ):
        return (
            _embed_query(
                query="ignored, model and chunker are mocked", model_path=None
            ),
            mock_model,
        )


def test_embed_query_mean_pools_chunk_embeddings_then_normalizes():
    # chunks point along [1, 0], [1, 0] and [0, 1]: their mean is [2/3, 1/3], which
    # differs from the max ([1, 1]), the first chunk ([1, 0]) and the last ([0, 1])
    embedding, _ = _embed_query_with_chunk_embeddings(
        chunk_embeddings=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    np.testing.assert_allclose(embedding, [2 / 5**0.5, 1 / 5**0.5])
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_embed_query_of_a_single_chunk_is_that_chunks_embedding():
    embedding, _ = _embed_query_with_chunk_embeddings(chunk_embeddings=[[0.6, 0.8]])
    np.testing.assert_allclose(embedding, [0.6, 0.8])


def test_embed_query_encodes_every_prefixed_chunk_normalized():
    _, mock_model = _embed_query_with_chunk_embeddings(
        chunk_embeddings=[[1.0, 0.0], [0.0, 1.0]]
    )
    (call_args, call_kwargs) = mock_model.encode.call_args
    assert call_args[0] == ["query: chunk 0", "query: chunk 1"]
    assert call_kwargs["normalize_embeddings"] is True


def test_embed_query_uses_the_fine_tuned_model_it_is_given():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0]])
    with (
        patch("lib.dense.retrieval.model", return_value=mock_model) as model_factory,
        patch(
            "lib.dense.retrieval.chunk_for_model",
            return_value=["a chunk"],
        ),
    ):
        _embed_query(query="ignored", model_path=Path("data/models/my-model"))
    model_factory.assert_called_once_with(path=Path("data/models/my-model"))
