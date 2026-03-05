"""
backend/tests/test_metrics.py

Tests for all IR evaluation metric functions.
"""

import pytest
from eval.metrics import mrr_at_k, hit_at_k, precision_at_k, recall_at_k, average_precision, compute_all


class TestMRRAtK:
    def test_first_position(self):
        assert mrr_at_k(["a", "b", "c"], ["a"], 3) == 1.0

    def test_second_position(self):
        assert mrr_at_k(["a", "b", "c"], ["b"], 3) == 0.5

    def test_third_position(self):
        assert mrr_at_k(["a", "b", "c"], ["c"], 3) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert mrr_at_k(["a", "b", "c"], ["z"], 3) == 0.0

    def test_k_limits_search(self):
        # "c" is at position 3, but k=2 so it's not found
        assert mrr_at_k(["a", "b", "c"], ["c"], 2) == 0.0

    def test_empty_retrieved(self):
        assert mrr_at_k([], ["a"], 5) == 0.0

    def test_empty_relevant(self):
        assert mrr_at_k(["a", "b"], [], 5) == 0.0


class TestHitAtK:
    def test_hit(self):
        assert hit_at_k(["a", "b", "c"], ["b"], 3) == 1.0

    def test_miss(self):
        assert hit_at_k(["a", "b", "c"], ["z"], 3) == 0.0

    def test_k_limits(self):
        assert hit_at_k(["a", "b", "c"], ["c"], 2) == 0.0

    def test_multiple_relevant(self):
        assert hit_at_k(["a", "b"], ["a", "b", "z"], 5) == 1.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b"], 3) == 0.0

    def test_partial(self):
        assert precision_at_k(["a", "x", "b"], ["a", "b"], 3) == pytest.approx(2 / 3)

    def test_zero_k(self):
        assert precision_at_k(["a"], ["a"], 0) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b"], 5) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], ["a", "b"], 5) == 0.5

    def test_no_relevant(self):
        assert recall_at_k(["a", "b"], [], 5) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        # All relevant items at top positions
        assert average_precision(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_single_relevant_at_top(self):
        assert average_precision(["a", "x", "y"], ["a"]) == 1.0

    def test_single_relevant_at_3rd(self):
        # precision at rank 3 = 1/3, ap = (1/3) / 1 = 1/3
        assert average_precision(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_empty_relevant(self):
        assert average_precision(["a", "b"], []) == 0.0


class TestComputeAll:
    def test_returns_all_keys(self):
        result = compute_all(["a", "b"], ["a"], k=5)
        assert set(result.keys()) == {"mrr", "hit", "precision", "recall", "ap"}

    def test_perfect_retrieval(self):
        result = compute_all(["a"], ["a"], k=5)
        assert result["mrr"] == 1.0
        assert result["hit"] == 1.0
        assert result["precision"] == pytest.approx(1 / 5)  # 1 hit in top-5
        assert result["recall"] == 1.0
