import numpy as np

from lib.finetuning.pair_mining import ChunkPair, best_chunk_pair, pairs_name


def test_best_chunk_pair_picks_the_highest_scoring_query_and_ruling_chunk():
    chunk_scores = [
        np.array([0.5, 2.0, 1.0]),  # query chunk 0: best is ruling chunk 1 (2.0)
        np.array([0.1, 0.2, 7.5]),  # query chunk 1: best is ruling chunk 2 (7.5)
    ]
    assert best_chunk_pair(chunk_scores=chunk_scores) == ChunkPair(
        query_chunk_index=1, ruling_chunk_index=2, score=7.5
    )


def test_best_chunk_pair_breaks_ties_toward_the_earliest_chunks():
    chunk_scores = [np.array([3.0, 3.0]), np.array([3.0, 3.0])]
    assert best_chunk_pair(chunk_scores=chunk_scores) == ChunkPair(
        query_chunk_index=0, ruling_chunk_index=0, score=3.0
    )


def test_best_chunk_pair_with_no_lexical_overlap_scores_zero():
    best = best_chunk_pair(chunk_scores=[np.zeros(3), np.zeros(3)])
    assert best == ChunkPair(query_chunk_index=0, ruling_chunk_index=0, score=0.0)


def test_pairs_name_identifies_the_sample_and_the_seed():
    assert pairs_name(train_cases=5000, seed=0) == "train_n5000_s0"
    assert pairs_name(train_cases=5000, seed=1) != pairs_name(train_cases=5000, seed=0)


def test_pairs_name_of_the_whole_train_split_ignores_the_seed():
    assert pairs_name(train_cases=None, seed=0) == "train"
    assert pairs_name(train_cases=None, seed=7) == "train"
