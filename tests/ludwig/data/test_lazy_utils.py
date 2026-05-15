"""Unit tests for LazyColumn.

Verifies that LazyColumn:
1. Decodes samples on demand (not upfront) — peak memory is bounded.
2. Returns correctly stacked numpy arrays for batch indices.
3. Handles all numpy index forms (int, list, slice, boolean mask).
4. Runs decode in parallel (ThreadPoolExecutor) and produces the same result as sequential decode.
5. Integrates transparently with PandasDataset.get().
"""

import numpy as np

from ludwig.data.lazy_utils import is_lazy_column, LazyColumn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paths(n: int) -> np.ndarray:
    """Return fake "paths" — just stringified integers for testing."""
    return np.array([str(i) for i in range(n)], dtype=object)


def _decode_int_path(path: str) -> np.ndarray:
    """Decode: parse int, return a (4,) float array filled with that int."""
    val = float(path)
    return np.full((4,), val, dtype=np.float32)


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------


class TestLazyColumnInterface:
    def setup_method(self):
        self.paths = _make_paths(10)
        self.col = LazyColumn(self.paths, _decode_int_path)

    def test_len(self):
        assert len(self.col) == 10

    def test_is_lazy_column(self):
        assert is_lazy_column(self.col)
        assert not is_lazy_column(np.zeros(10))

    def test_repr(self):
        assert "LazyColumn" in repr(self.col)
        assert "n=10" in repr(self.col)

    def test_shape(self):
        # Only batch dimension is known before decode
        assert self.col.shape == (10,)

    def test_dtype(self):
        assert self.col.dtype == object


class TestLazyColumnIndexing:
    def setup_method(self):
        self.paths = _make_paths(20)
        self.col = LazyColumn(self.paths, _decode_int_path)

    def test_integer_index(self):
        result = self.col[5]
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.full((4,), 5.0, dtype=np.float32))

    def test_list_of_indices(self):
        result = self.col[[0, 3, 7]]
        assert result.shape == (3, 4)
        np.testing.assert_array_almost_equal(result[:, 0], [0.0, 3.0, 7.0])

    def test_numpy_integer_array(self):
        idx = np.array([1, 2, 4])
        result = self.col[idx]
        assert result.shape == (3, 4)
        np.testing.assert_array_almost_equal(result[:, 0], [1.0, 2.0, 4.0])

    def test_slice(self):
        result = self.col[2:5]
        assert result.shape == (3, 4)
        np.testing.assert_array_almost_equal(result[:, 0], [2.0, 3.0, 4.0])

    def test_boolean_mask(self):
        mask = np.array([i % 2 == 0 for i in range(20)])
        result = self.col[mask]
        assert result.shape == (10, 4)
        expected = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=np.float32)
        np.testing.assert_array_almost_equal(result[:, 0], expected)

    def test_single_element_list(self):
        result = self.col[[3]]
        assert result.shape == (1, 4)


class TestLazyColumnCorrectness:
    """Verify decoded values match a reference sequential implementation."""

    def test_matches_sequential_decode(self):
        paths = _make_paths(50)
        col = LazyColumn(paths, _decode_int_path, max_workers=4)
        indices = list(range(50))
        result = col[indices]
        expected = np.stack([_decode_int_path(p) for p in paths[indices]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_worker_matches_multi(self):
        paths = _make_paths(30)
        col_1 = LazyColumn(paths, _decode_int_path, max_workers=1)
        col_4 = LazyColumn(paths, _decode_int_path, max_workers=4)
        idx = list(range(30))
        np.testing.assert_array_equal(col_1[idx], col_4[idx])

    def test_random_batch_selection(self):
        rng = np.random.default_rng(42)
        paths = _make_paths(100)
        col = LazyColumn(paths, _decode_int_path)
        idx = rng.choice(100, size=16, replace=False)
        result = col[idx]
        expected = np.stack([_decode_int_path(p) for p in paths[idx]])
        np.testing.assert_array_almost_equal(result, expected)


class TestLazyColumnPandasDatasetIntegration:
    """Test that LazyColumn integrates with PandasDataset.get()."""

    def test_get_returns_numpy_array(self):
        """PandasDataset.get() calls dataset[col][indices] — works for LazyColumn."""
        paths = _make_paths(32)
        col = LazyColumn(paths, _decode_int_path)
        # Simulate what PandasDataset.get() does:
        idx = [0, 5, 10, 31]
        result = col[idx]
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4)

    def test_decode_is_lazy(self):
        """Verify decode function is not called during construction."""
        call_count = {"n": 0}

        def counting_decode(path: str) -> np.ndarray:
            call_count["n"] += 1
            return np.zeros(4, dtype=np.float32)

        paths = _make_paths(100)
        col = LazyColumn(paths, counting_decode)
        # Construction must not have called decode
        assert call_count["n"] == 0
        # Accessing 5 samples triggers exactly 5 decode calls
        _ = col[[0, 1, 2, 3, 4]]
        assert call_count["n"] == 5
