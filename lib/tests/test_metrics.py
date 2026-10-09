import math

import pytest

from lib.metrics import most_common_baseline, mrr, top_k_ndcg, top_k_recall


class TestTopKRecall:
    def test_perfect_recall(self):
        assert top_k_recall(retrieved=["a", "b", "c"], gold={"a", "b", "c"}, k=3) == 1.0

    def test_partial_recall(self):
        assert top_k_recall(retrieved=["a", "x", "y"], gold={"a", "b"}, k=3) == 0.5

    def test_zero_recall(self):
        assert top_k_recall(retrieved=["x", "y", "z"], gold={"a", "b"}, k=3) == 0.0

    def test_k_smaller_than_retrieved(self):
        # relevant item is at rank 3, k=2 cuts it off
        assert top_k_recall(retrieved=["x", "y", "a"], gold={"a"}, k=2) == 0.0

    def test_k_larger_than_retrieved(self):
        assert top_k_recall(retrieved=["a"], gold={"a", "b"}, k=100) == 0.5

    def test_empty_gold_raises(self):
        with pytest.raises(ValueError):
            top_k_recall(retrieved=["a"], gold=set(), k=1)


class TestNdcgAtK:
    def test_perfect_ranking(self):
        assert top_k_ndcg(retrieved=["a", "b"], gold={"a", "b"}, k=2) == 1.0

    def test_relevant_item_ranked_lower_scores_worse(self):
        # single relevant item "a": ranking it first beats ranking it last
        first = top_k_ndcg(retrieved=["a", "x", "y"], gold={"a"}, k=3)
        last = top_k_ndcg(retrieved=["x", "y", "a"], gold={"a"}, k=3)
        assert last < first
        assert first == 1.0

    def test_no_relevant_items_retrieved(self):
        assert top_k_ndcg(retrieved=["x", "y"], gold={"a"}, k=2) == 0.0

    def test_matches_hand_computed_value(self):
        # gold = {a, b}, retrieved = [x, a, b] at k=3
        # DCG = 0/log2(2) + 1/log2(3) + 1/log2(4) = 1/1.585 + 1/2
        # IDCG (2 relevant, k=3) = 1/log2(2) + 1/log2(3) = 1 + 1/1.585
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert top_k_ndcg(
            retrieved=["x", "a", "b"], gold={"a", "b"}, k=3
        ) == pytest.approx(dcg / idcg)

    def test_empty_gold_raises(self):
        with pytest.raises(ValueError):
            top_k_ndcg(retrieved=["a"], gold=set(), k=1)


class TestMrr:
    def test_first_rank_hit(self):
        assert mrr(retrieved=["a", "x", "y"], gold={"a"}) == 1.0

    def test_third_rank_hit(self):
        assert mrr(retrieved=["x", "y", "a"], gold={"a"}) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert mrr(retrieved=["x", "y", "z"], gold={"a"}) == 0.0

    def test_uses_first_of_multiple_relevant_hits(self):
        assert mrr(retrieved=["x", "a", "b"], gold={"a", "b"}) == pytest.approx(1 / 2)

    def test_empty_gold_raises(self):
        with pytest.raises(ValueError):
            mrr(retrieved=["a"], gold=set())


class TestMostCommonBaseline:
    def test_returns_k_most_frequent(self):
        gold_sets = [{"a", "b"}, {"a"}, {"a", "c"}, {"b"}]
        assert most_common_baseline(gold_sets=gold_sets, k=2) == ["a", "b"]

    def test_k_larger_than_distinct_items(self):
        gold_sets = [{"a"}, {"b"}]
        assert set(most_common_baseline(gold_sets=gold_sets, k=10)) == {"a", "b"}

    def test_empty_input(self):
        assert most_common_baseline(gold_sets=[], k=5) == []
