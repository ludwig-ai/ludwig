import logging
import os
import time
from typing import Dict, List, Optional, Union

from torch.utils.tensorboard import SummaryWriter

from ludwig.constants import MODEL_LLM, TEST, TRAIN, TRAINING, VALIDATION
from ludwig.data.dataset.base import Dataset
from ludwig.distributed.base import DistributedStrategy, LocalStrategy
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.models.llm import LLM
from ludwig.models.predictor import Predictor
from ludwig.modules.metric_modules import get_initial_validation_value
from ludwig.schema.trainer import BaseTrainerConfig, ZeroShotTrainerConfig
from ludwig.trainers.base import BaseTrainer
from ludwig.trainers.registry import register_trainer
from ludwig.types import ModelConfigDict
from ludwig.utils import time_utils
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.metrics_printed_table import MetricsPrintedTable
from ludwig.utils.trainer_utils import ProgressTracker, append_metrics, get_new_progress_tracker


logger = logging.getLogger(__name__)


@register_trainer(MODEL_LLM)
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
        self.model = model
        self.eval_batch_size = 1
        self.base_learning_rate = 0.0
        self.distributed = distributed if distributed is not None else LocalStrategy()
        self.skip_save_log = skip_save_log
        self.callbacks = callbacks or []
        self.report_tqdm_to_ray = report_tqdm_to_ray

    def train(self, training_set, validation_set=None, test_set=None, save_path="model", **kwargs):
        output_features = self.model.output_features
        progress_tracker = get_new_progress_tracker(
            batch_size=-1,
            learning_rate=self.base_learning_rate,
            best_eval_metric_value=get_initial_validation_value(self.validation_metric),
            best_increase_batch_size_eval_metric=float("inf"),
            output_features=output_features,
        )

        # ====== Setup file names =======
        tensorboard_log_dir = None
        if self.is_coordinator():
            os.makedirs(save_path, exist_ok=True)
            tensorboard_log_dir = os.path.join(save_path, "logs")

        self.callback(
            lambda c: c.on_trainer_train_setup(self, save_path, self.is_coordinator()), coordinator_only=False
        )

        train_summary_writer = None
        validation_summary_writer = None
        test_summary_writer = None
        if self.is_coordinator() and not self.skip_save_log and tensorboard_log_dir:
            train_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, TRAINING))
            if validation_set is not None and validation_set.size > 0:
                validation_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, VALIDATION))
            if test_set is not None and test_set.size > 0:
                test_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, TEST))

        self.run_evaluation(
            training_set,
            validation_set,
            test_set,
            progress_tracker,
            train_summary_writer,
            validation_summary_writer,
            test_summary_writer,
            output_features,
            save_path,
        )

        return (
            self.model,
            progress_tracker.train_metrics,
            progress_tracker.validation_metrics,
            progress_tracker.test_metrics,
        )

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

    def is_coordinator(self) -> bool:
        return self.distributed.rank() == 0

    def callback(self, fn, coordinator_only=True):
        if not coordinator_only or self.is_coordinator():
            for callback in self.callbacks:
                fn(callback)

    def evaluation(
        self,
        dataset: "Dataset",  # noqa: F821
        dataset_name: str,
        metrics_log: Dict[str, Dict[str, List[TrainerMetric]]],
        batch_size: int,
        progress_tracker: ProgressTracker,
    ):
        predictor = Predictor(
            self.model, batch_size=batch_size, distributed=self.distributed, report_tqdm_to_ray=self.report_tqdm_to_ray
        )
        metrics, _ = predictor.batch_evaluation(dataset, collect_predictions=False, dataset_name=dataset_name)

        return append_metrics(self.model, dataset_name, metrics, metrics_log, progress_tracker)

    @classmethod
    def write_eval_summary(
        cls,
        summary_writer,
        metrics,
        step,
    ):
        if not summary_writer:
            return

        for feature_name, output_feature in metrics.items():
            for metric_name, metrics in output_feature.items():
                if metrics:
                    metric_tag = f"{feature_name}/epoch_{metric_name}"
                    metric_val = metrics[-1][-1]
                    summary_writer.add_scalar(metric_tag, metric_val, global_step=step)
        summary_writer.flush()

    def run_evaluation(
        self,
        training_set: Union["Dataset", "RayDataset"],  # noqa: F821
        validation_set: Optional[Union["Dataset", "RayDataset"]],  # noqa: F821
        test_set: Optional[Union["Dataset", "RayDataset"]],  # noqa: F821
        progress_tracker: ProgressTracker,
        train_summary_writer: SummaryWriter,
        validation_summary_writer: SummaryWriter,
        test_summary_writer: SummaryWriter,
        output_features: LudwigFeatureDict,
        save_path: str,
    ) -> bool:
        """Runs evaluation over training, validation, and test sets.

        Also:
        - Prints results, saves results to the progress tracker.
        - Saves the model if the validation score is the best so far
        - If there is no validation set, the model is always saved.

        Returns whether the trainer should early stop, based on validation metrics history.
        """
        start_time = time.time()
        self.callback(lambda c: c.on_eval_start(self, progress_tracker, save_path))

        progress_tracker.checkpoint_number += 1
        if self.is_coordinator():
            logger.info(f"\nRunning evaluation for step: {progress_tracker.steps}, epoch: {progress_tracker.epoch}")

        # ================ Eval ================
        printed_table = MetricsPrintedTable(output_features)

        # Run a separate pass over the training data to compute metrics
        # Appends results to progress_tracker.train_metrics.
        self.evaluation(training_set, "train", progress_tracker.train_metrics, self.eval_batch_size, progress_tracker)

        printed_table.add_metrics_to_printed_table(progress_tracker.train_metrics, TRAIN)

        self.write_eval_summary(
            summary_writer=train_summary_writer,
            metrics=progress_tracker.train_metrics,
            step=progress_tracker.steps,
        )

        if validation_set is not None:
            self.callback(lambda c: c.on_validation_start(self, progress_tracker, save_path))

            # eval metrics on validation set
            validation_metrics_log = self.evaluation(
                validation_set,
                VALIDATION,
                progress_tracker.validation_metrics,
                self.eval_batch_size,
                progress_tracker,
            )

            printed_table.add_metrics_to_printed_table(validation_metrics_log, VALIDATION)

            self.write_eval_summary(
                summary_writer=validation_summary_writer,
                metrics=progress_tracker.validation_metrics,
                step=progress_tracker.steps,
            )

            self.callback(lambda c: c.on_validation_end(self, progress_tracker, save_path))

        if test_set is not None:
            self.callback(lambda c: c.on_test_start(self, progress_tracker, save_path))

            # eval metrics on test set
            test_metrics_log = self.evaluation(
                test_set, TEST, progress_tracker.test_metrics, self.eval_batch_size, progress_tracker
            )

            printed_table.add_metrics_to_printed_table(test_metrics_log, TEST)

            self.write_eval_summary(
                summary_writer=test_summary_writer,
                metrics=progress_tracker.test_metrics,
                step=progress_tracker.steps,
            )

            self.callback(lambda c: c.on_test_end(self, progress_tracker, save_path))

        elapsed_time = (time.time() - start_time) * 1000.0

        if self.is_coordinator():
            logger.info(f"Evaluation took {time_utils.strdelta(elapsed_time)}\n")
            printed_table.log_info()

        # Trigger eval end callback after any model weights save for complete checkpoint
        self.callback(lambda c: c.on_eval_end(self, progress_tracker, save_path))

        return False
