from lib.fusion import reciprocal_rank_fusion


def test_an_item_ranked_high_by_both_methods_beats_one_ranked_first_by_only_one():
    fused = reciprocal_rank_fusion(rankings=[["a", "b", "c"], ["c", "b", "d"]], k=4)
    # c: ranks 3 and 1; b: ranks 2 and 2; a: rank 1 in one list only
    assert fused == ["c", "b", "a", "d"]


def test_k_truncates_the_fused_list():
    fused = reciprocal_rank_fusion(rankings=[["a", "b"], ["c", "d"]], k=3)
    assert len(fused) == 3


def test_ties_keep_the_order_of_first_appearance():
    assert reciprocal_rank_fusion(rankings=[["a"], ["b"]], k=2) == ["a", "b"]


def test_a_single_ranking_is_returned_unchanged():
    assert reciprocal_rank_fusion(rankings=[["x", "y", "z"]], k=3) == ["x", "y", "z"]
