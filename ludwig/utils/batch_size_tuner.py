import gc
import logging
import statistics
import time
from abc import ABC

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MAX_BATCH_SIZE_DATASET_FRACTION, MIN_POSSIBLE_BATCH_SIZE

logger = logging.getLogger(__name__)

TOTAL_STEPS = 5


@DeveloperAPI
class BatchSizeEvaluator(ABC):
    def select_best_batch_size(
        self,
        dataset_len: int,
        max_batch_size: int | None = None,
        max_trials: int = 20,
        is_coordinator: bool | None = True,
        global_max_sequence_length: int | None = None,
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
                samples_per_sec = self.evaluate(
                    batch_size, total_steps=TOTAL_STEPS, global_max_sequence_length=global_max_sequence_length
                )
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

    def evaluate(self, batch_size: int, total_steps: int = 5, global_max_sequence_length: int | None = None) -> float:
        """Evaluates throughput of the given batch size.

        Return:
            Median throughput in samples / sec.
        """
        durations = []
        for _ in range(total_steps):
            self.reset()
            start_ts = time.time()
            self.step(batch_size, global_max_sequence_length=global_max_sequence_length)
            durations.append(time.time() - start_ts)

        med_duration_s = statistics.median(durations)
        if med_duration_s == 0.0:
            return float("inf")

        return batch_size / med_duration_s

    def reset(self):
        """Called at the beginning of each evaluation step."""

    def step(self, batch_size: int, global_max_sequence_length: int | None = None):
        """Called each step to evaluate the given batch size."""
        raise NotImplementedError("`step` must be implemented by concrete evaluator.")


class BaseLLMBatchSizeEvaluator(BatchSizeEvaluator):
    """Base class for batch size evaluators for LLM models."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.input_feature_name, self.input_feature = list(trainer.model.input_features.items())[0]
        self.output_feature_name, self.output_feature = list(trainer.model.output_features.items())[0]

        # Get the length of the longest input sequence from the training data
        self.input_msl = self.input_feature.input_shape[0]
        if trainer.model.config_obj.input_features[0].preprocessing.max_sequence_length:
            self.input_msl = trainer.model.config_obj.input_features[0].preprocessing.max_sequence_length

        # Get the length of the longest output sequence from the training data
        self.output_msl = self.output_feature.output_shape[0]
        if trainer.model.config_obj.output_features[0].preprocessing.max_sequence_length:
            self.output_msl = trainer.model.config_obj.output_features[0].preprocessing.max_sequence_length

        # This is useful to create the synthetic input and target data which will be a
        # random sequence of integers between 0 and vocab_size
        self.vocab_size = len(trainer.model.config_obj.input_features[0].encoder.vocab)

    def reset(self):
        self.trainer.model.reset_metrics()
        self.trainer.optimizer.zero_grad()

    def step(self, batch_size: int, global_max_sequence_length: int | None = None):
        if global_max_sequence_length and self.input_msl + self.output_msl > global_max_sequence_length:
            # In this case, we just need to make sure that the length of the synthetic data exceeds
            # max_sequence_length by at most a small amount
            self.input_msl = global_max_sequence_length // 2 + 1
            self.output_msl = global_max_sequence_length // 2 + 1

        inputs = {
            self.input_feature_name: torch.randint(0, self.vocab_size, size=(batch_size, self.input_msl))
            .to(self.input_feature.input_dtype)
            .to(self.trainer.device)
        }
        targets = {
            self.output_feature_name: torch.randint(0, self.vocab_size, size=(batch_size, self.output_msl))
            .to(self.output_feature.get_output_dtype())
            .to(self.trainer.device)
        }

        self.perform_step(inputs, targets)

    def perform_step(self, inputs, targets):
        raise NotImplementedError("perform_step method must be implemented in subclasses")


class LLMFinetuneTrainerBatchSizeEvaluator(BaseLLMBatchSizeEvaluator):
    """Batch size evaluator for training batch size for LLM finetuning."""

    def perform_step(self, inputs, targets):
        self.trainer.train_step(inputs, targets)


class LLMFinetunePredictBatchSizeEvaluator(BaseLLMBatchSizeEvaluator):
    """Batch size evaluator for prediction/evaluation batch size for LLM finetuning."""

    def perform_step(self, inputs, targets):
        with torch.no_grad():
            self.trainer.dist_model((inputs, targets))
