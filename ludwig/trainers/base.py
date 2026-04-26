from abc import ABC, abstractmethod
from typing import Any

from ludwig.data.dataset.base import Dataset
from ludwig.globals import MODEL_FILE_NAME
from ludwig.schema.trainer import BaseTrainerConfig
from ludwig.types import ModelConfigDict
from ludwig.utils.defaults import default_random_seed


class BaseTrainer(ABC):
    """Abstract base class for all Ludwig trainers.

    Required methods (must be implemented by every subclass):
        train(), train_online(), tune_batch_size(), validation_field,
        validation_metric, get_schema_cls()

    Optional methods (have sensible no-op defaults; override as needed):
        shutdown(), barrier(), local_rank

    Use the `capabilities` class property to advertise non-standard features
    (e.g. {"distributed": True, "batch_size_tuning": False}) so callers can
    check support without catching NotImplementedError.
    """

    # Subclasses may override this dict to advertise capabilities.
    capabilities: dict[str, Any] = {}

    @abstractmethod
    def train(self, training_set, validation_set=None, test_set=None, save_path=MODEL_FILE_NAME, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def train_online(
        self,
        dataset,
    ):
        raise NotImplementedError()

    @abstractmethod
    def tune_batch_size(
        self,
        config: ModelConfigDict,
        training_set: Dataset,
        random_seed: int = default_random_seed,
        max_trials: int = 10,
        halving_limit: int = 3,
        tune_for_training: bool = True,
    ) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def validation_field(self) -> str:
        """Name of the output feature used for validation (e.g. "label")."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def validation_metric(self) -> str:
        """Name of the metric tracked on validation_field (e.g. "accuracy")."""
        raise NotImplementedError()

    # --- Optional methods (no-op defaults) ---

    def shutdown(self):
        """Release any resources held by the trainer.

        Called on context manager exit.
        """
        pass

    @property
    def local_rank(self) -> int:
        """Rank of this worker within the local node (0 for single-process trainers)."""
        return 0

    def barrier(self):
        """Synchronise all workers.

        No-op for single-process trainers.
        """
        pass

    # Context manager support

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    # --- Abstract class helpers ---

    @staticmethod
    @abstractmethod
    def get_schema_cls() -> BaseTrainerConfig:
        raise NotImplementedError()
