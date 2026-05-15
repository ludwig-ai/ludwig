"""Lazy media column utilities for Ludwig's preprocessing pipeline.

``LazyColumn`` wraps an array of file paths together with a per-sample decode
function.  It exposes the same ``__getitem__`` / ``__len__`` interface as a
numpy array, so ``PandasDataset.get()`` works transparently — callers never
need to know whether data was decoded eagerly or lazily.

Decoding happens inside a ``ThreadPoolExecutor`` at batch-slice time, which
means:

* Only ``batch_size`` samples are ever in memory simultaneously.
* CPU decode runs in parallel with GPU forward pass (pipeline overlap).
* Throughput matches the existing eager ``read_binary_files`` path because
  both use the same thread-pool approach.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

# Cap at 16 workers; beyond this point thread-switching overhead dominates
# for typical audio/image decode workloads.
_DEFAULT_MAX_WORKERS = min(16, (os.cpu_count() or 4) + 4)


class LazyColumn:
    """Array-like wrapper that decodes file paths on demand per batch.

    Parameters
    ----------
    paths:
        1-D array (or list) of file paths (strings) or raw bytes objects.
    decode_fn:
        Callable that takes a single path/bytes and returns a numpy array.
        Must be thread-safe (stateless or using only thread-local state).
    max_workers:
        Number of threads to use for parallel decode.  Defaults to
        ``min(16, cpu_count + 4)`` to match Python's ThreadPoolExecutor
        default policy.
    """

    def __init__(
        self,
        paths: np.ndarray | list,
        decode_fn: Callable[[str], np.ndarray],
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        self._paths = np.asarray(paths, dtype=object)
        self._decode_fn = decode_fn
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # numpy-compatible interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, indices) -> np.ndarray:
        """Decode the requested samples in parallel and return a numpy array.

        ``indices`` may be an integer, a list of ints, a numpy integer array,
        a boolean mask, or a Python ``slice`` — matching numpy semantics.
        """
        selected = self._paths[indices]

        # Scalar index (int or 0-d array) → wrap so map always sees an iterable
        scalar = isinstance(indices, (int, np.integer)) or (isinstance(selected, np.ndarray) and selected.ndim == 0)
        if scalar:
            selected = np.array([selected], dtype=object)

        paths_list = selected.tolist()

        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(paths_list))) as executor:
            decoded = list(executor.map(self._decode_fn, paths_list))

        result = np.stack(decoded)
        if scalar:
            return result[0]
        return result

    @property
    def dtype(self):
        return object  # paths are strings; callers that check dtype see 'object'

    @property
    def shape(self):
        # Only the batch dimension is known; sample shape requires a decode.
        return (len(self._paths),)

    def __repr__(self) -> str:
        return f"LazyColumn(n={len(self._paths)}, decode_fn={self._decode_fn.__name__!r})"


def is_lazy_column(col) -> bool:
    """Return True if *col* is a ``LazyColumn`` instance."""
    return isinstance(col, LazyColumn)
