from abc import ABC, abstractmethod

from ludwig.data.dataset.base import Dataset
from ludwig.schema.trainer import BaseTrainerConfig
from ludwig.types import ModelConfigDict
from ludwig.utils.defaults import default_random_seed


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, training_set, validation_set=None, test_set=None, save_path="model", **kwargs):
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
    ) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def validation_field(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def validation_metric(self):
        raise NotImplementedError()

    # Remote implementations may override this
    def shutdown(self):
        pass

    @property
    def local_rank(self) -> int:
        return 0

    def barrier(self):
        pass

    # Functions needed to treat Trainer as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @staticmethod
    @abstractmethod
    def get_schema_cls() -> BaseTrainerConfig:
        raise NotImplementedError()
