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
every partition completes.

Pandas/modin (non-Ray): the counter is an in-process integer; progress is fired
*synchronously* from increment() so every map_partitions call produces a callback
with no polling delay.

Dask/Ray: the counter is a Ray named actor so increments from remote workers are
visible on the head node.  A background thread polls the counter and fires
on_preprocess_progress(fraction) callbacks.  stop() drains the actor's queue
before emitting the final 1.0 so that increments fired during persist() are
always reflected.
"""

import threading
import time
from typing import Any


class _LocalProgressCounter:
    """In-process counter used by the pandas/modin backends."""

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

    Pandas/modin: callbacks fire synchronously on every ``increment()`` call --
    no background thread, no polling delay.

    Ray/Dask: a background thread polls the Ray actor at ``_POLL_INTERVAL_S``
    intervals and fires callbacks.  ``stop()`` drains the actor queue so that
    increments fired inside ``persist()`` are counted before the final 1.0 is
    emitted.
    """

    _POLL_INTERVAL_S = 0.5

    def __init__(self, total: int, callbacks: list, use_ray: bool = False):
        self._total = total
        self._callbacks = callbacks or []
        self._use_ray = use_ray

        if use_ray:
            actor = _make_ray_actor(total)
            self._counter = _RayProgressCounter(actor)
            self._actor = actor
            self._thread: threading.Thread | None = None
            self._stop_event = threading.Event()
        else:
            self._counter = _LocalProgressCounter(total)
            self._actor = None
            self._thread = None
            self._stop_event = threading.Event()

    def start(self):
        if self._use_ray:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        # pandas/modin: no thread needed; increment() fires synchronously.

    def stop(self):
        if self._use_ray and self._actor is not None:
            import ray

            # Drain the actor's call queue.  Ray actors serialize calls in
            # submission order, so issuing a get_completed() here guarantees
            # all previously submitted increment() calls have been applied
            # before we read the final count.  Since persist() has already
            # completed by the time stop() is called, this round-trip is
            # effectively instantaneous.
            try:
                ray.get(self._actor.get_completed.remote(), timeout=30)
            except Exception:
                pass  # best-effort; the final _fire(1.0) below covers it

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

        self._fire(1.0)

        if self._actor is not None:
            import ray

            ray.kill(self._actor)

    def increment(self):
        """Called from inside each map_partitions wrapper (in-process or remote)."""
        self._counter.increment()
        if not self._use_ray:
            # Synchronous fire for pandas/modin: every map_partitions call
            # immediately produces a progress update.
            completed = self._counter.completed
            if self._total > 0:
                self._fire(min(completed / self._total, 1.0))

    def get_actor(self) -> Any:
        """Returns the raw Ray actor so remote workers can call .increment.remote()."""
        return self._actor

    def _poll_loop(self):
        while not self._stop_event.is_set():
            completed = self._counter.completed
            if self._total > 0:
                self._fire(min(completed / self._total, 1.0))
            if completed >= self._total:
                break
            time.sleep(self._POLL_INTERVAL_S)

    def _fire(self, progress: float):
        for cb in self._callbacks:
            try:
                cb.on_preprocess_progress(progress=progress)
            except Exception:
                pass


def get_total_partitions(input_cols: dict, use_ray: bool) -> int:
    """Returns the number of partitions per feature column.

    For pandas/modin each column is a single partition (value=1).
    For Dask/Ray, all columns share the same partition scheme after
    repartitioning, so we read npartitions from the first column.
    """
    if not input_cols:
        return 1
    sample = next(iter(input_cols.values()))
    if use_ray and hasattr(sample, "npartitions"):
        return sample.npartitions
    return 1
