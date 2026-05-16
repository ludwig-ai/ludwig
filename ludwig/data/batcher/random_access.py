#! /usr/bin/env python
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
import logging
import math
import queue
import threading

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.data.batcher.base import Batcher

logger = logging.getLogger(__name__)


@DeveloperAPI
class RandomAccessBatcher(Batcher):
    """Batcher for a PandasDataset (local, non-distributed training).

    When ``prefetch_size > 0``, a background producer thread decodes the next
    ``prefetch_size`` batches concurrently with GPU work, so the GPU is never
    stalled waiting for disk I/O or compute decoding.  This is equivalent to
    the ``RayDatasetBatcher._create_async_reader`` pattern used in the Ray
    backend.

    ``prefetch_size=0`` (the default) uses the original synchronous path,
    which is appropriate for eager (pre-decoded) datasets where ``dataset.get``
    is a cheap numpy slice.

    ``PandasDataset.initialize_batcher`` automatically sets ``prefetch_size``
    to a non-zero value when any lazy column (audio/image path arrays) is
    present, so callers don't need to set this manually.
    """

    def __init__(
        self,
        dataset,
        sampler,
        batch_size: int = 128,
        ignore_last: bool = False,
        augmentation_pipeline=None,
        prefetch_size: int = 0,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.ignore_last = ignore_last
        self.batch_size = batch_size
        self.total_size = len(sampler)
        self.augmentation_pipeline = augmentation_pipeline
        self.steps_per_epoch = self._compute_steps_per_epoch()
        self._prefetch_size = prefetch_size

        # Mutable state consumed by both the sync and async paths.
        # When prefetch is active, ONLY the producer thread modifies these
        # (index / step / sample_it); the main thread only dequeues.
        self.index = 0
        self.step = 0
        self.sample_it = iter(self.sampler)

        # Async-path state — unused when prefetch_size == 0.
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_stop: threading.Event | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._async_last = False
        self._async_next = None

        if prefetch_size > 0:
            self._start_async_epoch()

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _sync_exhausted(self) -> bool:
        """Return True if there are no more batches for this epoch."""
        if self.index >= self.total_size:
            return True
        if self.ignore_last and self.step and self.batch_size > 1 and self.index - self.total_size == -1:
            logger.info("Last batch in epoch only has 1 sample and will be dropped.")
            return True
        return False

    def _fetch_sync(self) -> dict:
        """Fetch and decode one batch.  Only called from the producer thread when prefetch is on."""
        indices = []
        for _ in range(self.batch_size):
            try:
                indices.append(next(self.sample_it))
                self.index += 1
            except StopIteration:
                break

        sub_batch = {feature_name: self.dataset.get(feature_name, indices) for feature_name in self.dataset.features}

        if self.augmentation_pipeline:
            for feature_name, augmentations in self.augmentation_pipeline.items():
                logger.debug(f"RandomAccessBatcher applying augmentation pipeline to batch for feature {feature_name}")
                sub_batch[feature_name] = augmentations(torch.tensor(sub_batch[feature_name]))

        self.step += 1
        return sub_batch

    # ------------------------------------------------------------------
    # Async (prefetch) epoch management
    # ------------------------------------------------------------------

    def _start_async_epoch(self) -> None:
        """Spin up a fresh producer thread for one epoch."""
        stop = threading.Event()
        q = queue.Queue(maxsize=self._prefetch_size)
        self._prefetch_stop = stop
        self._prefetch_queue = q
        self._async_last = False
        self._async_next = None

        def producer():
            try:
                while not stop.is_set() and not self._sync_exhausted():
                    batch = self._fetch_sync()
                    # Blocking put with periodic stop-event checks so we don't
                    # hang forever if the consumer stops early (e.g. set_epoch).
                    while not stop.is_set():
                        try:
                            q.put(batch, timeout=0.2)
                            break
                        except queue.Full:
                            continue
            finally:
                # Sentinel is only useful for the normal-exhaustion case.
                # When stopped externally, _stop_async drains the queue before
                # this put, so there is always a free slot.
                if not stop.is_set():
                    q.put(None)
                else:
                    # Best-effort: there is always at least one free slot after
                    # _stop_async drains, so this won't block.
                    try:
                        q.put(None, block=False)
                    except queue.Full:
                        pass

        self._prefetch_thread = threading.Thread(target=producer, daemon=True)
        self._prefetch_thread.start()
        # Pre-load first value so last_batch() is accurate before the first
        # call to next_batch().
        self._async_advance()

    def _async_advance(self) -> None:
        """Dequeue the next pre-fetched batch; update _async_last on sentinel."""
        item = self._prefetch_queue.get()
        if item is None:
            self._async_last = True
            self._async_next = None
        else:
            self._async_last = False
            self._async_next = item

    def _stop_async(self) -> None:
        """Signal the producer to stop and wait for it to exit cleanly."""
        if self._prefetch_stop is not None:
            self._prefetch_stop.set()
        # Drain so the producer unblocks from q.put() on its next timeout,
        # and so the sentinel put in finally always has a free slot.
        if self._prefetch_queue is not None:
            while True:
                try:
                    self._prefetch_queue.get_nowait()
                except queue.Empty:
                    break
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
        self._prefetch_thread = None
        self._prefetch_stop = None
        self._prefetch_queue = None

    # ------------------------------------------------------------------
    # Public Batcher interface
    # ------------------------------------------------------------------

    def next_batch(self) -> dict:
        if self.last_batch():
            raise StopIteration()

        if self._prefetch_size > 0:
            # Fast path: return the already-decoded batch and kick off advance
            # of the next one (which overlaps with GPU work on the returned batch).
            batch = self._async_next
            self._async_advance()
            return batch

        # Sync path — unchanged from original; used for eager (pre-decoded) datasets.
        indices = []
        for _ in range(self.batch_size):
            try:
                indices.append(next(self.sample_it))
                self.index += 1
            except StopIteration:
                break

        sub_batch = {feature_name: self.dataset.get(feature_name, indices) for feature_name in self.dataset.features}

        if self.augmentation_pipeline:
            for feature_name, augmentations in self.augmentation_pipeline.items():
                logger.debug(f"RandomAccessBatcher applying augmentation pipeline to batch for feature {feature_name}")
                sub_batch[feature_name] = augmentations(torch.tensor(sub_batch[feature_name]))

        self.step += 1
        return sub_batch

    def last_batch(self) -> bool:
        """Returns whether we've exhausted all batches for this epoch."""
        if self._prefetch_size > 0:
            return self._async_last

        if self.index >= self.total_size:
            return True
        elif self.ignore_last and self.step:
            if self.batch_size > 1 and self.index - self.total_size == -1:
                logger.info("Last batch in epoch only has 1 sample and will be dropped.")
                return True
        return False

    def set_epoch(self, epoch: int, batch_size: int) -> None:
        if self._prefetch_size > 0:
            self._stop_async()

        self.batch_size = batch_size
        self.steps_per_epoch = self._compute_steps_per_epoch()
        self.index = 0
        self.step = 0
        self.sampler.set_epoch(epoch)
        self.sample_it = iter(self.sampler)

        if self._prefetch_size > 0:
            self._start_async_epoch()

    def _compute_steps_per_epoch(self) -> int:
        return int(math.ceil(self.total_size / self.batch_size))
