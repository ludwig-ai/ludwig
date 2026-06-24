# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Partition-level progress tracking for the preprocessing pipeline.

Each df-engine wraps its map_partitions call to increment a shared counter after
every partition completes.  On the local/modin backends the counter is a plain
in-process object; on the Dask/Ray backend it is a Ray named actor so increments
from remote workers are visible on the head node.  A lightweight background thread
polls the counter and fires on_preprocess_progress(fraction) callbacks.
"""

import threading
import time
from typing import Any


class _LocalProgressCounter:
    """Simple in-process counter used by the pandas/modin backends."""

    def __init__(self, total: int):
        self.total = total
        self._completed = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._completed += 1

    @property
    def completed(self) -> int:
        with self._lock:
            return self._completed


class _RayProgressCounter:
    """Head-node proxy around a Ray named actor."""

    def __init__(self, actor):
        self._actor = actor

    def increment(self):
        self._actor.increment.remote()

    @property
    def completed(self) -> int:
        import ray

        return ray.get(self._actor.get_completed.remote())

    @property
    def total(self) -> int:
        import ray

        return ray.get(self._actor.get_total.remote())


def _make_ray_actor(total: int):
    import ray

    @ray.remote
    class _ProgressActor:
        def __init__(self, t: int):
            self._completed = 0
            self._total = t

        def increment(self):
            self._completed += 1

        def get_completed(self) -> int:
            return self._completed

        def get_total(self) -> int:
            return self._total

    return _ProgressActor.remote(total)


class PreprocessingProgressTracker:
    """Fires ``on_preprocess_progress`` callbacks as partitions complete.

    Usage::

        tracker = PreprocessingProgressTracker(total_partitions, callbacks, use_ray=False)
        tracker.start()
        # pass tracker.increment to each map_partitions call
        ...
        tracker.stop()
    """

    _POLL_INTERVAL_S = 0.5

    def __init__(self, total: int, callbacks: list, use_ray: bool = False):
        if use_ray:
            actor = _make_ray_actor(total)
            self._counter = _RayProgressCounter(actor)
            self._actor = actor
        else:
            self._counter = _LocalProgressCounter(total)
            self._actor = None

        self._total = total
        self._callbacks = callbacks or []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # Fire a final 1.0 so callers always see completion.
        self._fire(1.0)
        if self._actor is not None:
            import ray

            ray.kill(self._actor)

    def increment(self):
        """Called from inside each partition function (in-process or remote)."""
        self._counter.increment()

    def get_actor(self) -> Any:
        """Returns the raw Ray actor so remote workers can call .increment.remote()."""
        return self._actor

    def _poll_loop(self):
        while not self._stop_event.is_set():
            completed = self._counter.completed
            total = self._total
            if total > 0:
                self._fire(min(completed / total, 1.0))
            if completed >= total:
                break
            time.sleep(self._POLL_INTERVAL_S)

    def _fire(self, progress: float):
        for cb in self._callbacks:
            try:
                cb.on_preprocess_progress(progress=progress)
            except Exception:
                pass


def get_total_partitions(input_cols: dict, use_ray: bool) -> int:
    """Returns the number of partitions across all feature columns.

    For pandas/modin each column is 1 partition.  For Dask each column has
    npartitions partitions; we take the max since all columns share the same
    partition scheme after repartitioning.
    """
    if not input_cols:
        return 1
    sample = next(iter(input_cols.values()))
    if use_ray and hasattr(sample, "npartitions"):
        # All Dask columns share the same npartitions after repartition.
        return sample.npartitions
    return 1
