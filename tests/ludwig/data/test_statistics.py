"""Unit tests for WelfordAccumulator (online mean/variance/min/max).

These tests ensure:
1. Single-pass statistics match numpy batch statistics exactly (within floating point).
2. Shard-merge is commutative and associative.
3. Edge cases (single sample, constant series, negative values) are handled correctly.
"""

import math

import numpy as np

from ludwig.data.statistics import welford_from_array, welford_stats_match_numpy, WelfordAccumulator


class TestWelfordSingleSeries:
    def test_matches_numpy_uniform(self):
        xs = np.random.default_rng(0).uniform(-10, 10, size=1000)
        acc = welford_from_array(xs)
        assert welford_stats_match_numpy(acc, xs)

    def test_matches_numpy_normal(self):
        xs = np.random.default_rng(1).normal(5.0, 2.0, size=500)
        acc = welford_from_array(xs)
        assert welford_stats_match_numpy(acc, xs)

    def test_matches_numpy_single_element(self):
        acc = WelfordAccumulator()
        acc.update(42.0)
        assert acc.count == 1
        assert math.isclose(float(acc.mean), 42.0)
        assert float(acc.std) == 0.0
        assert math.isclose(float(acc.min), 42.0)
        assert math.isclose(float(acc.max), 42.0)

    def test_constant_series_zero_variance(self):
        acc = WelfordAccumulator()
        for _ in range(100):
            acc.update(7.0)
        assert math.isclose(float(acc.mean), 7.0)
        assert float(acc.std) < 1e-10
        assert float(acc.variance) < 1e-10

    def test_min_max(self):
        xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        acc = welford_from_array(xs)
        assert math.isclose(float(acc.min), 1.0)
        assert math.isclose(float(acc.max), 9.0)

    def test_negative_values(self):
        xs = np.linspace(-100, -1, 50)
        acc = welford_from_array(xs)
        assert welford_stats_match_numpy(acc, xs)

    def test_result_dict_keys(self):
        acc = welford_from_array([1.0, 2.0, 3.0])
        result = acc.result()
        assert set(result.keys()) == {"count", "mean", "std", "variance", "min", "max"}
        assert result["count"] == 3

    def test_update_batch(self):
        xs = np.arange(1.0, 101.0)
        acc = WelfordAccumulator()
        acc.update_batch(xs)
        assert welford_stats_match_numpy(acc, xs)


class TestWelfordMerge:
    def test_merge_two_equal_shards(self):
        rng = np.random.default_rng(42)
        xs = rng.normal(0, 1, size=200)
        half = len(xs) // 2
        acc_a = welford_from_array(xs[:half])
        acc_b = welford_from_array(xs[half:])
        merged = WelfordAccumulator.merge(acc_a, acc_b)
        assert welford_stats_match_numpy(merged, xs)

    def test_merge_unequal_shards(self):
        rng = np.random.default_rng(7)
        xs = rng.exponential(2.0, size=300)
        acc_a = welford_from_array(xs[:50])
        acc_b = welford_from_array(xs[50:])
        merged = WelfordAccumulator.merge(acc_a, acc_b)
        assert welford_stats_match_numpy(merged, xs)

    def test_merge_with_empty(self):
        xs = [1.0, 2.0, 3.0]
        acc = welford_from_array(xs)
        empty = WelfordAccumulator()
        assert WelfordAccumulator.merge(acc, empty).count == 3
        assert WelfordAccumulator.merge(empty, acc).count == 3

    def test_merge_all(self):
        rng = np.random.default_rng(99)
        xs = rng.standard_normal(600)
        shards = [welford_from_array(xs[i * 100 : (i + 1) * 100]) for i in range(6)]
        merged = WelfordAccumulator.merge_all(shards)
        assert welford_stats_match_numpy(merged, xs)

    def test_merge_commutativity(self):
        xs1 = np.array([1.0, 2.0, 3.0])
        xs2 = np.array([4.0, 5.0])
        ab = WelfordAccumulator.merge(welford_from_array(xs1), welford_from_array(xs2))
        ba = WelfordAccumulator.merge(welford_from_array(xs2), welford_from_array(xs1))
        assert math.isclose(float(ab.mean), float(ba.mean))
        assert math.isclose(float(ab.std), float(ba.std))

    def test_merge_associativity(self):
        a = welford_from_array([1.0, 2.0])
        b = welford_from_array([3.0, 4.0])
        c = welford_from_array([5.0])
        ab_c = WelfordAccumulator.merge(WelfordAccumulator.merge(a, b), c)
        a_bc = WelfordAccumulator.merge(a, WelfordAccumulator.merge(b, c))
        assert math.isclose(float(ab_c.mean), float(a_bc.mean), rel_tol=1e-12)
        assert math.isclose(float(ab_c.std), float(a_bc.std), rel_tol=1e-12)


class TestWelfordEdgeCases:
    def test_empty_accumulator_defaults(self):
        acc = WelfordAccumulator()
        assert acc.count == 0
        assert float(acc.mean) == 0.0
        assert float(acc.std) == 0.0

    def test_large_values(self):
        xs = np.array([1e15, 2e15, 3e15])
        acc = welford_from_array(xs)
        assert welford_stats_match_numpy(acc, xs, rtol=1e-6)

    def test_two_elements(self):
        acc = WelfordAccumulator()
        acc.update(0.0)
        acc.update(2.0)
        assert math.isclose(float(acc.mean), 1.0)
        assert math.isclose(float(acc.std), math.sqrt(2.0))
