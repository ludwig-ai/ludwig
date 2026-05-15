"""Online statistics accumulator using Welford's algorithm.

Supports O(1) memory accumulation of mean, variance, min, max, and count
over arbitrarily large datasets without materialising the full data.

Also supports merging accumulators from independent shards, making it
safe to use in distributed / multi-worker settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WelfordAccumulator:
    """Incremental mean/variance accumulator (Welford's online algorithm).

    Numerically stable for large N.  All statistics are maintained per
    scalar dimension: pass a 1-D numpy array to ``update`` and every
    element is tracked independently (useful for per-channel image stats).

    Usage::

        acc = WelfordAccumulator()
        for batch in data_loader:
            acc.update(batch)          # batch: 1-D np.ndarray of floats
        mean, std = acc.mean, acc.std  # scalar or per-element arrays

    Merging shards::

        acc = WelfordAccumulator.merge_all([acc_shard_0, acc_shard_1, ...])
    """

    count: int = 0
    _mean: np.ndarray = field(default_factory=lambda: np.array(0.0))
    # Welford's M2 accumulator: sum of squared deviations from the running mean
    _m2: np.ndarray = field(default_factory=lambda: np.array(0.0))
    _min: np.ndarray = field(default_factory=lambda: np.array(np.inf))
    _max: np.ndarray = field(default_factory=lambda: np.array(-np.inf))

    def update(self, x: float | np.ndarray) -> None:
        """Incorporate a new observation (scalar or 1-D array)."""
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = x - self._mean
        self._mean = self._mean + delta / self.count
        delta2 = x - self._mean
        self._m2 = self._m2 + delta * delta2
        self._min = np.minimum(self._min, x)
        self._max = np.maximum(self._max, x)

    def update_batch(self, xs: np.ndarray) -> None:
        """Incorporate a batch of scalar observations efficiently.

        ``xs`` shape: (N,) — one scalar per sample.
        """
        xs = np.asarray(xs, dtype=np.float64).ravel()
        for x in xs:
            self.update(x)

    @property
    def mean(self) -> np.ndarray:
        return self._mean if self.count > 0 else np.array(0.0)

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.array(0.0)
        return self._m2 / (self.count - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    @property
    def min(self) -> np.ndarray:
        return self._min if self.count > 0 else np.array(np.nan)

    @property
    def max(self) -> np.ndarray:
        return self._max if self.count > 0 else np.array(np.nan)

    def result(self) -> dict:
        """Return all statistics as a plain dict."""
        return {
            "count": self.count,
            "mean": float(self.mean),
            "std": float(self.std),
            "variance": float(self.variance),
            "min": float(self.min),
            "max": float(self.max),
        }

    @classmethod
    def merge(cls, a: WelfordAccumulator, b: WelfordAccumulator) -> WelfordAccumulator:
        """Parallel / Chan's algorithm merge of two independent accumulators.

        The result is mathematically equivalent to having accumulated all
        samples from both accumulators in a single pass.
        """
        if a.count == 0:
            return b
        if b.count == 0:
            return a

        merged = cls()
        merged.count = a.count + b.count
        delta = b._mean - a._mean
        merged._mean = (a._mean * a.count + b._mean * b.count) / merged.count
        merged._m2 = a._m2 + b._m2 + delta**2 * a.count * b.count / merged.count
        merged._min = np.minimum(a._min, b._min)
        merged._max = np.maximum(a._max, b._max)
        return merged

    @classmethod
    def merge_all(cls, accumulators: list[WelfordAccumulator]) -> WelfordAccumulator:
        """Reduce a list of accumulators into one."""
        result = cls()
        for acc in accumulators:
            result = cls.merge(result, acc)
        return result


def welford_from_array(xs: np.ndarray) -> WelfordAccumulator:
    """Convenience: build an accumulator from a 1-D array in one call."""
    acc = WelfordAccumulator()
    for x in np.asarray(xs, dtype=np.float64).ravel():
        acc.update(x)
    return acc


def welford_stats_match_numpy(acc: WelfordAccumulator, xs: np.ndarray, rtol: float = 1e-6) -> bool:
    """Return True if Welford stats agree with numpy reference within rtol."""
    xs = np.asarray(xs, dtype=np.float64).ravel()
    np_mean = float(np.mean(xs))
    np_std = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    np_min = float(np.min(xs))
    np_max = float(np.max(xs))

    return (
        math.isclose(float(acc.mean), np_mean, rel_tol=rtol)
        and math.isclose(float(acc.std), np_std, rel_tol=rtol)
        and math.isclose(float(acc.min), np_min, rel_tol=rtol)
        and math.isclose(float(acc.max), np_max, rel_tol=rtol)
    )
