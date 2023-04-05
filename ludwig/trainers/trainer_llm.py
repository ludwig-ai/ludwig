from typing import List, Optional
from ludwig.data.dataset.base import Dataset
from ludwig.distributed.base import DistributedStrategy
from ludwig.models.llm import LLM
from ludwig.schema.trainer import BaseTrainerConfig, ZeroShotTrainerConfig
from ludwig.trainers.base import BaseTrainer
from ludwig.types import ModelConfigDict
from ludwig.utils.defaults import default_random_seed


class ZeroShotTrainer(BaseTrainer):
    """ZeroShotTrainer is a trainer that does not train a model."""

    def __init__(
        self,
        config: ZeroShotTrainerConfig,
        model: LLM,
        resume: float = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        callbacks: List = None,
        report_tqdm_to_ray=False,
        random_seed: float = default_random_seed,
        distributed: Optional[DistributedStrategy] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config

    def train(self, training_set, validation_set=None, test_set=None, save_path="model", **kwargs):
        pass

    def train_online(
        self,
        dataset,
    ):
        pass

    def tune_batch_size(
        self,
        config: ModelConfigDict,
        training_set: Dataset,
        random_seed: int = default_random_seed,
        max_trials: int = 10,
        halving_limit: int = 3,
    ) -> int:
        return 1

    @property
    def validation_field(self):
        return self.config.validation_field

    @property
    def validation_metric(self):
        return self.config.validation_metric

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
    def get_schema_cls() -> BaseTrainerConfig:
        return ZeroShotTrainerConfig
