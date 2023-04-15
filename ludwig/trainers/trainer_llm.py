import logging
import os
import sys
import time
from typing import Optional, Union

from torch.utils.tensorboard import SummaryWriter

from ludwig.constants import MODEL_LLM, TEST, TRAIN, TRAINING, VALIDATION
from ludwig.data.dataset.base import Dataset
from ludwig.distributed.base import DistributedStrategy, LocalStrategy
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.globals import is_progressbar_disabled
from ludwig.models.llm import LLM
from ludwig.models.predictor import Predictor
from ludwig.modules.metric_modules import get_initial_validation_value
from ludwig.progress_bar import LudwigProgressBar
from ludwig.schema.trainer import BaseTrainerConfig, ZeroShotTrainerConfig
from ludwig.trainers.base import BaseTrainer
from ludwig.trainers.registry import register_ray_trainer, register_trainer
from ludwig.types import ModelConfigDict
from ludwig.utils import time_utils
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.metrics_printed_table import MetricsPrintedTable
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.torch_utils import get_torch_device
from ludwig.utils.trainer_utils import (
    append_metrics,
    get_final_steps_per_checkpoint,
    get_new_progress_tracker,
    get_total_steps,
    ProgressTracker,
)

logger = logging.getLogger(__name__)


@register_ray_trainer(MODEL_LLM)
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
        callbacks: list = None,
        report_tqdm_to_ray=False,
        random_seed: float = default_random_seed,
        distributed: Optional[DistributedStrategy] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        :param config: `ludwig.schema.trainer.ZeroShotTrainerConfig` instance that specifies training hyperparameters
        (default: `ludwig.schema.trainer.ZeroShotTrainerConfig()`).
        :param model: Underlying Ludwig model
        :type model: `ludwig.models.llm.LLM`
        :param resume: Resume training a model that was being trained. (default: False).
        :type resume: Boolean
        :param skip_save_model: Disables saving model weights and hyperparameters each time the model improves. By
                default Ludwig saves model weights after each round of evaluation the validation metric (improves, but
                if the model is really big that can be time consuming. If you do not want to keep the weights and just
                find out what performance a model can get with a set of hyperparameters, use this parameter to skip it,
                but the model will not be loadable later on. (default: False).
        :type skip_save_model: Boolean
        :param skip_save_progress: Disables saving progress each round of evaluation. By default Ludwig saves weights
                and stats after each round of evaluation for enabling resuming of training, but if the model is really
                big that can be time consuming and will uses twice as much space, use this parameter to skip it, but
                training cannot be resumed later on. (default: False).
        :type skip_save_progress: Boolean
        :param skip_save_log: Disables saving TensorBoard logs. By default Ludwig saves logs for the TensorBoard, but if
                it is not needed turning it off can slightly increase the overall speed. (default: False).
        :type skip_save_log: Boolean
        :param callbacks: List of `ludwig.callbacks.Callback` objects that provide hooks into the Ludwig pipeline.
                (default: None).
        :type callbacks: list
        :param report_tqdm_to_ray: Enables using the ray based tqdm Callback for progress bar reporting
        :param random_seed: Default initialization for the random seeds (default: 42).
        :type random_seed: Float
        :param distributed: Distributed strategy (default: None).
        :type distributed: `DistributedStrategy`
        :param device: Device to load the model on from a saved checkpoint (default: None).
        :type device: str
        """

        super().__init__()
        self.config = config
        self.distributed = distributed if distributed is not None else LocalStrategy()
        self.skip_save_log = skip_save_log
        self.resume = resume
        self.skip_save_model = skip_save_model
        self.skip_save_progress = skip_save_progress
        self.random_seed = random_seed
        self.device = device
        self.callbacks = callbacks or []
        self.report_tqdm_to_ray = report_tqdm_to_ray

        if self.device is None:
            self.device = get_torch_device()

        self.model = model
        self.model = self.model.to(self.device)

        self.batch_size = self.config.batch_size
        self.eval_batch_size = self.config.eval_batch_size
        self.base_learning_rate = self.config.base_learning_rate
        self.should_shuffle = self.config.should_shuffle
        self.epochs = self.config.epochs
        self.train_steps = self.config.train_steps
        self.steps_per_checkpoint = self.config.steps_per_checkpoint
        self.checkpoints_per_epoch = self.config.checkpoints_per_epoch
        self.early_stop = self.config.early_stop

    def train(self, training_set, validation_set=None, test_set=None, save_path="model", **kwargs):
        logger.info("Starting Training")
        output_features = self.model.output_features

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

        set_random_seed(self.random_seed)

        progress_tracker = get_new_progress_tracker(
            batch_size=self.batch_size,
            learning_rate=self.base_learning_rate,
            best_eval_metric_value=get_initial_validation_value(self.validation_metric),
            best_increase_batch_size_eval_metric=float("inf"),
            output_features=output_features,
        )

        try:
            with training_set.initialize_batcher(
                batch_size=self.batch_size,
                should_shuffle=self.should_shuffle,
                random_seed=self.random_seed,
                distributed=self.distributed,
                ignore_last=True,
                augmentation_pipeline=self.model.get_augmentation_pipelines(),
            ) as batcher:
                # ================ Training Loop ================
                self.total_steps = get_total_steps(self.epochs, batcher.steps_per_epoch, self.train_steps)

                # Get the terminal steps per checkpoint.
                final_steps_per_checkpoint = get_final_steps_per_checkpoint(
                    batcher.steps_per_epoch,
                    self.steps_per_checkpoint,
                    self.checkpoints_per_epoch,
                    self.is_coordinator(),
                )
                final_steps_per_checkpoint = min(final_steps_per_checkpoint, self.total_steps)
                early_stopping_steps = final_steps_per_checkpoint * self.early_stop

                if self.is_coordinator():
                    logger.info(
                        f"Training for {self.total_steps} step(s), approximately "
                        f"{int(self.total_steps / batcher.steps_per_epoch)} epoch(s)."
                    )
                    if self.early_stop < 0:
                        logger.info("Early stopping policy: None")
                    else:
                        logger.info(
                            f"Early stopping policy: {self.early_stop} round(s) of evaluation, or "
                            f"{early_stopping_steps} step(s), approximately "
                            f"{int(early_stopping_steps / batcher.steps_per_epoch)} epoch(s).\n"
                        )
                    logger.info(f"Starting with step {progress_tracker.steps}, epoch: {progress_tracker.epoch}")

                progress_bar_config = {
                    "desc": "Training",
                    "total": self.total_steps,
                    "disable": is_progressbar_disabled(),
                    "file": sys.stdout,
                }
                progress_bar = LudwigProgressBar(self.report_tqdm_to_ray, progress_bar_config, self.is_coordinator())

                while progress_tracker.steps < self.total_steps:
                    # note that batch size may change over epochs
                    batcher.set_epoch(progress_tracker.epoch, progress_tracker.batch_size)

                    # epoch init
                    start_time = time.time()

                    # Reset the metrics at the start of the next epoch
                    self.model.reset_metrics()

                    self.callback(lambda c: c.on_epoch_start(self, progress_tracker, save_path))

                    # Trains over a full epoch of data.
                    should_break = self._train_loop(
                        batcher,
                        progress_tracker,
                        save_path,
                        train_summary_writer,
                        progress_bar,
                        training_set,
                        validation_set,
                        test_set,
                        start_time,
                        validation_summary_writer,
                        test_summary_writer,
                        output_features,
                        final_steps_per_checkpoint,
                        early_stopping_steps,
                    )

                    # ================ Post Training Epoch ================
                    progress_tracker.epoch += 1
                    self.callback(lambda c: c.on_epoch_end(self, progress_tracker, save_path))

                    if self.is_coordinator():
                        # ========== Save training progress ==========
                        logger.debug(
                            f"Epoch {progress_tracker.epoch} took: "
                            f"{time_utils.strdelta((time.time() - start_time) * 1000.0)}."
                        )

                    # Early stop if needed.
                    if should_break:
                        break

        finally:
            # ================ Finished Training ================
            self.callback(
                lambda c: c.on_trainer_train_teardown(self, progress_tracker, save_path, self.is_coordinator()),
                coordinator_only=False,
            )

            if train_summary_writer is not None:
                train_summary_writer.close()
            if validation_summary_writer is not None:
                validation_summary_writer.close()
            if test_summary_writer is not None:
                test_summary_writer.close()

            return (
                self.model,
                progress_tracker.train_metrics,
                progress_tracker.validation_metrics,
                progress_tracker.test_metrics,
            )

    def _train_loop(
        self,
        batcher,
        progress_tracker,
        save_path,
        train_summary_writer,
        progress_bar,
        training_set,
        validation_set,
        test_set,
        start_time,
        validation_summary_writer,
        test_summary_writer,
        output_features,
        final_steps_per_checkpoint: int,
        early_stopping_steps: int,
    ) -> bool:
        """Completes up to one epoch through the data."""
        while not batcher.last_batch() and progress_tracker.steps < self.total_steps:
            self.callback(lambda c: c.on_batch_start(self, progress_tracker, save_path))

            progress_tracker.steps += 1
            progress_bar.update(1)

            # Executing `on_batch_end` calls before `run_evaluation` enables more accurate
            # batch duration measurements when using timer callbacks.
            self.callback(lambda c: c.on_batch_end(self, progress_tracker, save_path, sync_step=False))

            if progress_tracker.steps % final_steps_per_checkpoint == 0:
                should_break = self.run_evaluation(
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
                if should_break:
                    return should_break

        return False

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
        metrics_log: dict[str, dict[str, list[TrainerMetric]]],
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

        # eval metrics on the train set
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
