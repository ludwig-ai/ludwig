"""Tests for split utilities."""

import numpy as np

from ludwig.data.split_utils import get_split_indices, stratified_split_indices


class TestGetSplitIndices:
    def test_default_proportions(self):
        splits = get_split_indices(1000)
        train = (splits == 0).sum()
        val = (splits == 1).sum()
        test = (splits == 2).sum()
        assert abs(train - 700) < 10
        assert abs(val - 100) < 10
        assert abs(test - 200) < 10

    def test_custom_proportions(self):
        splits = get_split_indices(1000, probabilities=(0.8, 0.1, 0.1))
        train = (splits == 0).sum()
        assert abs(train - 800) < 10

    def test_reproducible(self):
        s1 = get_split_indices(100, random_seed=42)
        s2 = get_split_indices(100, random_seed=42)
        assert np.array_equal(s1, s2)

    def test_different_seeds(self):
        s1 = get_split_indices(100, random_seed=42)
        s2 = get_split_indices(100, random_seed=99)
        assert not np.array_equal(s1, s2)


class TestStratifiedSplitIndices:
    def test_maintains_distribution(self):
        labels = np.array([0] * 500 + [1] * 500)
        splits = stratified_split_indices(labels)

        train_0 = ((splits == 0) & (labels == 0)).sum()
        train_1 = ((splits == 0) & (labels == 1)).sum()
        # Both classes should have ~70% in training
        assert abs(train_0 - 350) < 20
        assert abs(train_1 - 350) < 20

    def test_imbalanced(self):
        labels = np.array([0] * 900 + [1] * 100)
        splits = stratified_split_indices(labels)

        # Minority class should still be split proportionally
        test_1 = ((splits == 2) & (labels == 1)).sum()
        assert test_1 >= 10  # At least some minority in test
