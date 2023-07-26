import gc
import logging
import statistics
import time
from abc import ABC
from typing import Optional

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MAX_BATCH_SIZE_DATASET_FRACTION, MIN_POSSIBLE_BATCH_SIZE

logger = logging.getLogger(__name__)


@DeveloperAPI
class BatchSizeEvaluator(ABC):
    def select_best_batch_size(
        self,
        dataset_len: int,
        max_batch_size: Optional[int] = None,
        max_trials: int = 20,
        is_coordinator: Optional[bool] = True,
    ) -> int:
        """Returns optimal batch size as measured by throughput (samples / sec)."""
        logger.info("Tuning batch size...")

        max_batch_size = max_batch_size or dataset_len

        def _is_valid_batch_size(batch_size):
            # make sure that batch size is valid (e.g. less than 20% of ds size and max_batch_size)
            is_smaller_than_training_set = batch_size <= MAX_BATCH_SIZE_DATASET_FRACTION * dataset_len
            is_under_max_batch_size = batch_size <= max_batch_size
            is_valid = is_smaller_than_training_set and is_under_max_batch_size
            if not is_valid and is_coordinator:
                logger.info(
                    f"Batch size {batch_size} is invalid, must be less than or equal to "
                    f"{MAX_BATCH_SIZE_DATASET_FRACTION * 100}% dataset size "
                    f"({int(MAX_BATCH_SIZE_DATASET_FRACTION * dataset_len)} samples "
                    f"of {dataset_len}) and less than or equal to max batch size {max_batch_size}"
                )
            return is_valid

        batch_size = MIN_POSSIBLE_BATCH_SIZE
        best_samples_per_sec = 0
        best_batch_size = None
        count = 0
        while count < max_trials and _is_valid_batch_size(batch_size):
            if is_coordinator:
                logger.info(f"Exploring batch_size={batch_size}")
            gc.collect()

            try:
                samples_per_sec = self.evaluate(batch_size, total_steps=5)
                if is_coordinator:
                    logger.info(f"Throughput at batch_size={batch_size}: {samples_per_sec:.5f} samples/s")
                if samples_per_sec < best_samples_per_sec:
                    # We assume that once the throughput starts degrading, it won't go up again
                    if is_coordinator:
                        logger.info(f"Throughput decrease at batch_size={batch_size}")
                    break

                best_samples_per_sec = samples_per_sec
                best_batch_size = batch_size
                count += 1

                # double batch size
                batch_size *= 2
            except RuntimeError as e:
                # PyTorch only generates Runtime errors for CUDA OOM.
                gc.collect()
                if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                    if is_coordinator:
                        logger.info(f"OOM at batch_size={batch_size}")
                else:
                    # Not a CUDA error
                    raise
                break

        # Ensure that some batch size is found.
        # `best_batch_size` can be None if the first batch size is invalid.
        if best_batch_size is None:
            if is_coordinator:
                logger.info(f"Could not tune batch size, using minimum batch size of {MIN_POSSIBLE_BATCH_SIZE}")
            best_batch_size = MIN_POSSIBLE_BATCH_SIZE

        if is_coordinator:
            logger.info(f"Selected batch_size={best_batch_size}")
        return best_batch_size

    def evaluate(self, batch_size: int, total_steps: int = 5) -> float:
        """Evaluates throughput of the given batch size.

        Return:
            Median throughput in samples / sec.
        """
        durations = []
        for _ in range(total_steps):
            self.reset()
            start_ts = time.time()
            self.step(batch_size)
            durations.append(time.time() - start_ts)

        med_duration_s = statistics.median(durations)
        if med_duration_s == 0.0:
            return float("inf")

        return batch_size / med_duration_s

    def reset(self):
        """Called at the beginning of each evaluation step."""
        pass

    def step(self, batch_size: int):
        """Called each step to evaluate the given batch size."""
        raise NotImplementedError("`step` must be implemented by concrete evaluator.")
