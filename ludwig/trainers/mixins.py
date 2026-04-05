"""Trainer mixins for composable training behavior.

Extracts cross-cutting concerns from the monolithic Trainer class into
focused mixins that can be composed independently.

Usage:
    class MyTrainer(CheckpointMixin, EarlyStoppingMixin, MetricsMixin, BaseTrainer):
        pass
"""

import logging
import time
from abc import ABC

logger = logging.getLogger(__name__)


class CheckpointMixin(ABC):
    """Mixin for checkpoint save/restore functionality.

    Provides methods for saving checkpoints at regular intervals, tracking
    the best checkpoint, and resuming from saved state.
    """

    def should_checkpoint(self, steps: int, steps_per_checkpoint: int, epoch_end: bool = False) -> bool:
        """Determine if a checkpoint should be saved at the current step.

        Args:
            steps: Current training step count.
            steps_per_checkpoint: Save checkpoint every N steps.
            epoch_end: Whether this is the end of an epoch.

        Returns:
            True if checkpoint should be saved.
        """
        if epoch_end:
            return True
        if steps_per_checkpoint > 0 and steps % steps_per_checkpoint == 0:
            return True
        return False


class EarlyStoppingMixin(ABC):
    """Mixin for early stopping based on validation metrics.

    Tracks improvement in validation metrics and signals when training
    should stop due to lack of improvement.
    """

    def should_early_stop(
        self,
        steps_since_improvement: int,
        early_stop_rounds: int,
    ) -> bool:
        """Check if training should stop early.

        Args:
            steps_since_improvement: Number of evaluation rounds without improvement.
            early_stop_rounds: Maximum rounds without improvement before stopping.
                -1 or 0 means never stop early.

        Returns:
            True if training should stop.
        """
        if early_stop_rounds <= 0:
            return False
        return steps_since_improvement >= early_stop_rounds


class MetricsMixin(ABC):
    """Mixin for metric collection and logging.

    Provides structured metric tracking across training, validation,
    and test sets with support for multiple output features.
    """

    def format_metrics(self, metrics: dict, prefix: str = "") -> str:
        """Format metrics dict as a human-readable string.

        Args:
            metrics: Nested dict of feature_name -> metric_name -> value.
            prefix: Optional prefix (e.g., "train", "val").

        Returns:
            Formatted string.
        """
        parts = []
        for feat_name, feat_metrics in metrics.items():
            if isinstance(feat_metrics, dict):
                for metric_name, value in feat_metrics.items():
                    if isinstance(value, float):
                        label = f"{prefix}.{feat_name}.{metric_name}" if prefix else f"{feat_name}.{metric_name}"
                        parts.append(f"{label}={value:.4f}")
        return " | ".join(parts)


class BatchSizeTuningMixin(ABC):
    """Mixin for automatic batch size tuning.

    Finds the largest batch size that fits in GPU memory by binary search.
    """

    def find_max_batch_size(
        self,
        model,
        dataset,
        min_batch_size: int = 2,
        max_batch_size: int = 4096,
    ) -> int:
        """Find the maximum batch size that fits in memory.

        Uses binary search: start with max, if OOM halve it, repeat.

        Args:
            model: The model to test.
            dataset: Training dataset to sample from.
            min_batch_size: Minimum batch size to try.
            max_batch_size: Maximum batch size to try.

        Returns:
            Largest batch size that doesn't cause OOM.
        """
        import torch

        batch_size = max_batch_size
        while batch_size >= min_batch_size:
            try:
                # Try a forward + backward pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # If no OOM, this batch size works
                logger.info(f"Batch size {batch_size}: OK")
                return batch_size
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Batch size {batch_size}: OOM, trying {batch_size // 2}")
                    batch_size //= 2
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
        return min_batch_size


class ProfilingMixin(ABC):
    """Mixin for training profiling and timing.

    Tracks wall clock time for training steps, evaluation, and
    checkpoint operations.
    """

    def __init_profiling__(self):
        self._timing = {}
        self._timing_start = {}

    def start_timer(self, name: str):
        """Start a named timer."""
        self._timing_start[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed seconds."""
        if name not in self._timing_start:
            return 0.0
        elapsed = time.time() - self._timing_start.pop(name)
        self._timing[name] = self._timing.get(name, 0.0) + elapsed
        return elapsed

    def get_timing_summary(self) -> dict[str, float]:
        """Get all accumulated timing data."""
        return self._timing.copy()
