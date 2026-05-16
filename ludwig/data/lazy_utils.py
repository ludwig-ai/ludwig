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
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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


def _select(paths: np.ndarray, indices) -> tuple[list, list, bool]:
    """Normalise any index type to (paths_list, int_indices_list, is_scalar).

    Accepts int, np.integer, list, np.ndarray, slice, and bool mask.
    Returns a flat list of paths, a flat list of integer indices, and a flag
    indicating whether the original index was scalar (so callers can unwrap).
    """
    scalar = isinstance(indices, (int, np.integer))
    if not scalar and isinstance(indices, np.ndarray) and indices.ndim == 0:
        scalar = True

    if scalar:
        idx = int(indices)
        return [paths[idx]], [idx], True

    if isinstance(indices, slice):
        int_indices = list(range(*indices.indices(len(paths))))
    elif isinstance(indices, np.ndarray) and indices.dtype == bool:
        int_indices = list(np.where(indices)[0])
    elif isinstance(indices, np.ndarray):
        int_indices = indices.tolist()
    else:
        int_indices = list(indices)

    return [paths[i] for i in int_indices], int_indices, False


class CachedLazyColumn:
    """Like :class:`LazyColumn`, but writes decoded arrays to a numpy memmap.

    On the first pass through the dataset every decoded batch is written to a
    ``np.memmap`` file alongside the Parquet cache.  Once every sample has been
    written (tracked by a bool array + a ``.done`` sentinel file), subsequent
    epochs read directly from the memmap — no decode, no thread pool.

    Parameters
    ----------
    paths:
        1-D array or list of file paths.
    decode_fn:
        Callable ``path -> np.ndarray`` (same contract as :class:`LazyColumn`).
    cache_path:
        Full path for the memmap file (e.g. ``/data/audio_proc_decoded_n1000_8_23_f32.npy``).
    sample_shape:
        Shape of a single decoded sample, e.g. ``(feature_dim, max_length)``.
    dtype:
        Element dtype for the memmap. Default ``np.float32``.
    max_workers:
        Thread-pool size for parallel decode on cache-miss. Default ``_DEFAULT_MAX_WORKERS``.
    """

    def __init__(
        self,
        paths: np.ndarray | list,
        decode_fn: Callable[[str], np.ndarray],
        cache_path: str,
        sample_shape: tuple,
        dtype=np.float32,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        self._paths = np.asarray(paths, dtype=object)
        self._decode_fn = decode_fn
        self._cache_path = cache_path
        self._done_path = cache_path + ".done"
        self._sample_shape = sample_shape
        self._dtype = dtype
        self._max_workers = max_workers
        self._n = len(self._paths)
        self._written = np.zeros(self._n, dtype=bool)
        self._lock = threading.Lock()
        self._memmap = None
        self._fully_cached = os.path.exists(self._done_path)
        if self._fully_cached:
            self._written[:] = True
            self._memmap = np.memmap(cache_path, dtype=dtype, mode="r", shape=(self._n, *sample_shape))

    # ------------------------------------------------------------------
    # numpy-compatible interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, indices) -> np.ndarray:
        _, int_indices, scalar = _select(self._paths, indices)

        if self._fully_cached:
            result = np.array(self._memmap[int_indices])
            return result[0] if scalar else result

        # Decode any samples not yet in the cache.
        need_decode = [i for i in int_indices if not self._written[i]]

        if need_decode:
            paths_list = [self._paths[i] for i in need_decode]
            with ThreadPoolExecutor(max_workers=min(self._max_workers, len(paths_list))) as ex:
                decoded = list(ex.map(self._decode_fn, paths_list))

            with self._lock:
                if self._memmap is None:
                    os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
                    self._memmap = np.memmap(
                        self._cache_path, dtype=self._dtype, mode="w+", shape=(self._n, *self._sample_shape)
                    )
                for i, arr in zip(need_decode, decoded):
                    if not self._written[i]:
                        self._memmap[i] = arr
                        self._written[i] = True

                if self._written.all() and not self._fully_cached:
                    self._fully_cached = True
                    self._memmap.flush()
                    Path(self._done_path).touch()

        result = np.array(self._memmap[int_indices])
        return result[0] if scalar else result

    @property
    def dtype(self):
        return object

    @property
    def shape(self):
        return (self._n,)

    def is_fully_cached(self) -> bool:
        """Return True once every sample has been decoded and written to the memmap."""
        return self._fully_cached

    def __repr__(self) -> str:
        return f"CachedLazyColumn(n={self._n}, cached={self._fully_cached})"


def is_lazy_column(col) -> bool:
    """Return True if *col* is a ``LazyColumn`` or ``CachedLazyColumn`` instance."""
    return isinstance(col, (LazyColumn, CachedLazyColumn))


def is_cached_lazy_column(col) -> bool:
    """Return True if *col* is a ``CachedLazyColumn`` instance."""
    return isinstance(col, CachedLazyColumn)


def get_default_lazy_cache_dir() -> Path:
    """Return the root directory used for lazy media caches.

    Creates ``~/.cache/ludwig/lazy_media/`` on first call if it does not
    already exist.  All per-feature subdirectories are nested inside this root.

    Returns
    -------
    Path
        Absolute path to the root cache directory.
    """
    cache_root = Path.home() / ".cache" / "ludwig" / "lazy_media"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def resolve_lazy_cache_dir(cache_dir_param: str | None, feature_name: str) -> Path:
    """Resolve and create the per-feature lazy cache directory.

    If *cache_dir_param* is given, it is used as the parent directory and
    *feature_name* is appended as a subdirectory.  When *cache_dir_param* is
    ``None``, the default root returned by :func:`get_default_lazy_cache_dir`
    is used instead.

    Parameters
    ----------
    cache_dir_param:
        Explicit cache directory string from the preprocessing config, or
        ``None`` to use the default location.
    feature_name:
        Name of the Ludwig feature (e.g. ``"audio"`` or ``"image"``).  Used as
        the leaf directory name so that multiple features do not share a cache.

    Returns
    -------
    Path
        Absolute path to the per-feature cache directory.  The directory is
        guaranteed to exist after this call.

    Examples
    --------
    >>> resolve_lazy_cache_dir(None, "my_audio")
    PosixPath('/home/user/.cache/ludwig/lazy_media/my_audio')
    >>> resolve_lazy_cache_dir("/tmp/my_cache", "my_image")
    PosixPath('/tmp/my_cache/my_image')
    """
    if cache_dir_param is not None:
        feature_cache_dir = Path(cache_dir_param) / feature_name
    else:
        feature_cache_dir = get_default_lazy_cache_dir() / feature_name
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    return feature_cache_dir
