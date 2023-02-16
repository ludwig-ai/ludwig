import logging
import os
import re
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import torch
from torch.utils.tensorboard import SummaryWriter

from ludwig.constants import BINARY, CATEGORY, MINIMIZE, MODEL_GBM, NUMBER, TEST, TRAIN, TRAINING, VALIDATION
from ludwig.distributed.base import DistributedStrategy, LocalStrategy
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.globals import is_progressbar_disabled, TRAINING_CHECKPOINTS_DIR_PATH, TRAINING_PROGRESS_TRACKER_FILE_NAME
from ludwig.models.gbm import GBM
from ludwig.models.predictor import Predictor
from ludwig.modules.metric_modules import get_improved_fn, get_initial_validation_value
from ludwig.modules.metric_registry import get_metric_objective
from ludwig.progress_bar import LudwigProgressBar
from ludwig.schema.trainer import BaseTrainerConfig, GBMTrainerConfig
from ludwig.trainers.base import BaseTrainer
from ludwig.trainers.registry import register_ray_trainer, register_trainer
from ludwig.types import ModelConfigDict
from ludwig.utils import time_utils
from ludwig.utils.checkpoint_utils import Checkpoint, CheckpointManager
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.gbm_utils import (
    get_single_output_feature,
    get_targets,
    log_loss_objective,
    logits_to_predictions,
    multiclass_objective,
    store_predictions,
    store_predictions_ray,
    TrainLogits,
)
from ludwig.utils.metric_utils import get_metric_names, TrainerMetric
from ludwig.utils.metrics_printed_table import MetricsPrintedTable
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.trainer_utils import (
    append_metrics,
    get_latest_metrics_dict,
    get_new_progress_tracker,
    ProgressTracker,
)

logger = logging.getLogger(__name__)


@register_trainer(MODEL_GBM)
class LightGBMTrainer(BaseTrainer):
    TRAIN_KEY = "train"
    VALID_KEY = "validation"
    TEST_KEY = "test"

    def __init__(
        self,
        config: GBMTrainerConfig,
        model: GBM,
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

        self.random_seed = random_seed
        self.model = model
        self.distributed = distributed if distributed is not None else LocalStrategy()
        self.received_sigint = False
        self.report_tqdm_to_ray = report_tqdm_to_ray
        self.callbacks = callbacks or []
        self.skip_save_progress = skip_save_progress
        self.skip_save_log = skip_save_log
        self.skip_save_model = skip_save_model

        self.eval_batch_size = config.eval_batch_size
        self._validation_field = config.validation_field
        self._validation_metric = config.validation_metric
        self.evaluate_training_set = config.evaluate_training_set
        try:
            base_learning_rate = float(config.learning_rate)
        except ValueError:
            base_learning_rate = 0.001  # Default initial learning rate for autoML.
        self.base_learning_rate = base_learning_rate
        self.early_stop = config.early_stop

        self.boosting_type = config.boosting_type
        self.tree_learner = config.tree_learner
        self.num_boost_round = config.num_boost_round
        self.boosting_rounds_per_checkpoint = min(self.num_boost_round, config.boosting_rounds_per_checkpoint)
        self.max_depth = config.max_depth
        self.num_leaves = config.num_leaves
        self.min_data_in_leaf = config.min_data_in_leaf
        self.min_sum_hessian_in_leaf = config.min_sum_hessian_in_leaf
        self.feature_fraction = config.feature_fraction
        self.bagging_fraction = config.bagging_fraction
        self.pos_bagging_fraction = config.pos_bagging_fraction
        self.neg_bagging_fraction = config.neg_bagging_fraction
        self.bagging_seed = config.bagging_seed
        self.bagging_freq = config.bagging_freq
        self.feature_fraction_bynode = config.feature_fraction_bynode
        self.feature_fraction_seed = config.feature_fraction_seed
        self.extra_trees = config.extra_trees
        self.extra_seed = config.extra_seed
        self.max_delta_step = config.max_delta_step
        self.lambda_l1 = config.lambda_l1
        self.lambda_l2 = config.lambda_l2
        self.linear_lambda = config.linear_lambda
        self.min_gain_to_split = config.min_gain_to_split
        self.drop_rate = config.drop_rate
        self.max_drop = config.max_drop
        self.skip_drop = config.skip_drop
        self.xgboost_dart_mode = config.xgboost_dart_mode
        self.uniform_drop = config.uniform_drop
        self.drop_seed = config.drop_seed
        self.top_rate = config.top_rate
        self.other_rate = config.other_rate
        self.min_data_per_group = config.min_data_per_group
        self.max_cat_threshold = config.max_cat_threshold
        self.cat_l2 = config.cat_l2
        self.cat_smooth = config.cat_smooth
        self.max_cat_to_onehot = config.max_cat_to_onehot
        self.cegb_tradeoff = config.cegb_tradeoff
        self.cegb_penalty_split = config.cegb_penalty_split
        self.path_smooth = config.path_smooth
        self.verbose = config.verbose
        self.max_bin = config.max_bin
        self.feature_pre_filter = config.feature_pre_filter

        if self.boosting_type == "goss" and self.bagging_freq != 0:
            logger.info(
                "Bagging is not compatible with the GOSS boosting type. Disabling bagging for this training session."
            )
            self.bagging_freq = 0

        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # when training starts the sigint handler will be replaced with
        # set_steps_to_1_or_quit so this is needed to remember
        # the original sigint to restore at the end of training
        # and before set_steps_to_1_or_quit returns
        self.original_sigint_handler = None

    @staticmethod
    def get_schema_cls() -> BaseTrainerConfig:
        return GBMTrainerConfig

    def tune_batch_size(
        self,
        config: ModelConfigDict,
        training_set: "Dataset",  # noqa: F821
        random_seed: int,
        max_trials: int = 10,
        halving_limit: int = 3,
    ) -> int:
        raise NotImplementedError("Tuning batch size is not supported for LightGBM.")

    def train_online(
        self,
        dataset,
    ):
        raise NotImplementedError("Online training is not supported for LightGBM.")

    @property
    def validation_field(self) -> str:
        return self._validation_field

    @property
    def validation_metric(self) -> str:
        return self._validation_metric

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
        metrics_names: Dict[str, List[str]],
        save_path: str,
        loss: torch.Tensor,
        all_losses: Dict[str, torch.Tensor],
        early_stopping_steps: int,
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

        # eval metrics on train
        if self.evaluate_training_set:
            # Run a separate pass over the training data to compute metrics
            # Appends results to progress_tracker.train_metrics.
            self.evaluation(
                training_set, "train", progress_tracker.train_metrics, self.eval_batch_size, progress_tracker
            )
        else:
            # Use metrics accumulated during training
            metrics = self.model.get_metrics()
            append_metrics(self.model, "train", metrics, progress_tracker.train_metrics, progress_tracker)
            self.model.reset_metrics()

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

        # ================ Validation Logic ================
        should_break = False
        if validation_set is not None and validation_set.size > 0:
            should_break = self.check_progress_on_validation(
                progress_tracker,
                self.validation_field,
                self.validation_metric,
                save_path,
                early_stopping_steps,
                self.skip_save_model,
            )
        else:
            # There's no validation, so we save the model.
            if self.is_coordinator() and not self.skip_save_model:
                self.model.save(save_path)

        # Trigger eval end callback after any model weights save for complete checkpoint
        self.callback(lambda c: c.on_eval_end(self, progress_tracker, save_path))

        return should_break

    def _train_loop(
        self,
        params: Dict[str, Any],
        lgb_train: lgb.Dataset,
        eval_sets: List[lgb.Dataset],
        eval_names: List[str],
        progress_tracker: ProgressTracker,
        progress_bar: LudwigProgressBar,
        save_path: str,
        training_set: Union["Dataset", "RayDataset"],  # noqa: F821
        validation_set: Union["Dataset", "RayDataset"],  # noqa: F821
        test_set: Union["Dataset", "RayDataset"],  # noqa: F821
        train_summary_writer: SummaryWriter,
        validation_summary_writer: SummaryWriter,
        test_summary_writer: SummaryWriter,
        early_stopping_steps: int,
    ) -> bool:
        self.callback(lambda c: c.on_batch_start(self, progress_tracker, save_path))

        evals_result = {}
        self.model.lgbm_model = self.train_step(
            params,
            lgb_train,
            eval_sets,
            eval_names,
            self.model.lgbm_model,
            self.boosting_rounds_per_checkpoint,
            evals_result,
        )

        progress_bar.update(self.boosting_rounds_per_checkpoint)
        progress_tracker.steps += self.boosting_rounds_per_checkpoint

        output_features = self.model.output_features
        metrics_names = get_metric_names(output_features)
        output_feature_name = next(iter(output_features))

        loss_name = params["metric"][0]
        loss = evals_result["train"][loss_name][-1]
        loss = torch.tensor(loss, dtype=torch.float32)

        should_break = self.run_evaluation(
            training_set,
            validation_set,
            test_set,
            progress_tracker,
            train_summary_writer,
            validation_summary_writer,
            test_summary_writer,
            output_features,
            metrics_names,
            save_path,
            loss,
            {output_feature_name: loss},
            early_stopping_steps,
        )

        self.callback(lambda c: c.on_batch_end(self, progress_tracker, save_path))

        return should_break

    def check_progress_on_validation(
        self,
        progress_tracker: ProgressTracker,
        validation_output_feature_name: str,
        validation_metric: str,
        save_path: str,
        early_stopping_steps: int,
        skip_save_model: bool,
    ) -> bool:
        """Checks the history of validation scores.

        Uses history of validation scores to decide whether training
        should stop.

        Saves the model if scores have improved.
        """
        should_break = False
        # record how long its been since an improvement
        improved = get_improved_fn(validation_metric)
        all_validation_metrics = progress_tracker.validation_metrics[validation_output_feature_name]
        # The most recent validation_metric metric.
        eval_metric: TrainerMetric = all_validation_metrics[validation_metric][-1]
        eval_metric_value = eval_metric[-1]

        if improved(eval_metric_value, progress_tracker.best_eval_metric_value):
            previous_best_eval_metric_value = progress_tracker.best_eval_metric_value

            # Save the value, steps, epoch, and checkpoint number.
            progress_tracker.best_eval_metric_value = eval_metric_value

            # Use LGBM fine-grained internal tracking if available, otherwise fall back to coarse-grained tracking
            # every `boosting_rounds_per_checkpoint`.
            if self.model.lgbm_model.best_iteration_ is not None:
                progress_tracker.best_eval_metric_steps = self.model.lgbm_model.best_iteration_
            else:
                progress_tracker.best_eval_metric_steps = progress_tracker.steps
            progress_tracker.best_eval_metric_epoch = progress_tracker.epoch
            progress_tracker.best_eval_metric_checkpoint_number = progress_tracker.checkpoint_number

            # Save best metrics for all data subsets.
            progress_tracker.best_eval_train_metrics = get_latest_metrics_dict(progress_tracker.train_metrics)
            progress_tracker.best_eval_validation_metrics = get_latest_metrics_dict(progress_tracker.validation_metrics)
            progress_tracker.best_eval_test_metrics = get_latest_metrics_dict(progress_tracker.test_metrics)

            if self.is_coordinator():
                logger.info(
                    f"Evaluation validation metric: '{validation_output_feature_name}' '{validation_metric}' improved."
                )
                absolute_eval_metric_value_change = round(
                    abs(previous_best_eval_metric_value - progress_tracker.best_eval_metric_value), 3
                )
                if get_metric_objective(validation_metric) == MINIMIZE:
                    logger.info(
                        f"'{validation_output_feature_name}' '{validation_metric}' decreased by "
                        f"{absolute_eval_metric_value_change}."
                    )
                else:
                    logger.info(
                        f"'{validation_output_feature_name}' '{validation_metric}' increased by "
                        f"{absolute_eval_metric_value_change}."
                    )
                # Save the model.
                if not skip_save_model:
                    logger.info("New best model saved.\n")
                    self.model.save(save_path)

        last_improvement_in_steps = progress_tracker.steps - progress_tracker.best_eval_metric_steps
        progress_tracker.last_improvement_steps = last_improvement_in_steps

        if last_improvement_in_steps != 0 and self.is_coordinator():
            logger.info(
                f"Last improvement of {validation_output_feature_name} validation {validation_metric} happened "
                + f"{last_improvement_in_steps} step(s) ago.\n"
            )

        # ========== Early Stop logic ==========
        # If any early stopping condition is satisfied, either lack of improvement for many steps, or via callbacks on
        # any worker, then trigger early stopping.
        early_stop_bool = 0 < early_stopping_steps <= last_improvement_in_steps
        if not early_stop_bool:
            for callback in self.callbacks:
                if callback.should_early_stop(self, progress_tracker, self.is_coordinator()):
                    early_stop_bool = True
                    break

        should_early_stop = torch.as_tensor([early_stop_bool], dtype=torch.int)
        should_early_stop = self.distributed.allreduce(should_early_stop)
        if should_early_stop.item():
            if self.is_coordinator():
                logger.info(
                    f"\nEARLY STOPPING due to lack of validation improvement. It has been {last_improvement_in_steps} "
                    "step(s) since last validation improvement."
                )
            should_break = True
        return should_break

    def train_step(
        self,
        params: Dict[str, Any],
        lgb_train: lgb.Dataset,
        eval_sets: List[lgb.Dataset],
        eval_names: List[str],
        init_model: lgb.LGBMModel,
        boost_rounds_per_train_step: int,
        evals_result: Dict,
    ) -> lgb.LGBMModel:
        """Trains a LightGBM model.

        Args:
            params: parameters for LightGBM
            lgb_train: LightGBM dataset for training
            eval_sets: LightGBM datasets for evaluation
            eval_names: names of the evaluation datasets

        Returns:
            LightGBM Booster model
        """
        output_feature = get_single_output_feature(self.model)
        is_regression = output_feature.type() == NUMBER
        gbm_sklearn_cls = lgb.LGBMRegressor if is_regression else lgb.LGBMClassifier

        # Prepare buffer for storing predictions on training data.
        train_logits = torch.zeros(
            (lgb_train.label.size, output_feature.num_classes)
            if output_feature.type() == CATEGORY and output_feature.num_classes > 2
            else (lgb_train.label.size,)
        )

        callbacks = [store_predictions(train_logits)]

        # DART is not compatible with early stopping.
        if self.boosting_type != "dart":
            callbacks.append(lgb.early_stopping(boost_rounds_per_train_step))

        gbm = gbm_sklearn_cls(n_estimators=boost_rounds_per_train_step, **params).fit(
            X=lgb_train.get_data(),
            y=lgb_train.get_label(),
            init_model=init_model,
            eval_set=[(ds.get_data(), ds.get_label()) for ds in eval_sets],
            eval_names=eval_names,
            # add early stopping callback to populate best_iteration
            callbacks=callbacks,
        )
        evals_result.update(gbm.evals_result_)

        if not self.evaluate_training_set:
            # Update evaluation metrics with current model params:
            # noisy but fast way to get metrics on the training set
            predictions = logits_to_predictions(self.model, train_logits)
            targets = get_targets(lgb_train, output_feature, self.device)
            self.model.update_metrics(targets, predictions)

        return gbm

    def train(
        self,
        training_set: Union["Dataset", "RayDataset"],  # noqa: F821
        validation_set: Optional[Union["Dataset", "RayDataset"]],  # noqa: F821
        test_set: Optional[Union["Dataset", "RayDataset"]],  # noqa: F821
        save_path="model",
        **kwargs,
    ):
        # ====== General setup =======
        output_features = self.model.output_features

        # Only use signals when on the main thread to avoid issues with CherryPy
        # https://github.com/ludwig-ai/ludwig/issues/286
        if threading.current_thread() == threading.main_thread():
            # set the original sigint signal handler
            # as we want to restore it at the end of training
            self.original_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self.set_steps_to_1_or_quit)

        # TODO: construct new datasets by running encoders (for text, image)

        # ====== Setup file names =======
        training_checkpoints_path = None
        tensorboard_log_dir = None
        if self.is_coordinator():
            os.makedirs(save_path, exist_ok=True)
            training_checkpoints_path = os.path.join(save_path, TRAINING_CHECKPOINTS_DIR_PATH)
            tensorboard_log_dir = os.path.join(save_path, "logs")

        self.callback(
            lambda c: c.on_trainer_train_setup(self, save_path, self.is_coordinator()), coordinator_only=False
        )

        # ====== Setup session =======
        checkpoint = checkpoint_manager = None
        if self.is_coordinator() and not self.skip_save_progress:
            checkpoint = Checkpoint(model=self.model)
            checkpoint_manager = CheckpointManager(checkpoint, training_checkpoints_path, device=self.device)

        train_summary_writer = None
        validation_summary_writer = None
        test_summary_writer = None
        if self.is_coordinator() and not self.skip_save_log and tensorboard_log_dir:
            train_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, TRAINING))
            if validation_set is not None and validation_set.size > 0:
                validation_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, VALIDATION))
            if test_set is not None and test_set.size > 0:
                test_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, TEST))

        progress_tracker = get_new_progress_tracker(
            batch_size=-1,
            learning_rate=self.base_learning_rate,
            best_eval_metric_value=get_initial_validation_value(self.validation_metric),
            best_increase_batch_size_eval_metric=float("inf"),
            output_features=output_features,
        )

        set_random_seed(self.random_seed)

        try:
            params = self._construct_lgb_params()

            lgb_train, eval_sets, eval_names = self._construct_lgb_datasets(training_set, validation_set, test_set)

            # use separate total steps variable to allow custom SIGINT logic
            self.total_steps = self.num_boost_round

            early_stopping_steps = self.boosting_rounds_per_checkpoint * self.early_stop

            if self.is_coordinator():
                logger.info(
                    f"Training for {self.total_steps} boosting round(s), approximately "
                    f"{int(self.total_steps / self.boosting_rounds_per_checkpoint)} round(s) of evaluation."
                )
                logger.info(
                    f"Early stopping policy: {self.early_stop} round(s) of evaluation, or {early_stopping_steps} "
                    f"boosting round(s).\n"
                )

                logger.info(f"Starting with step {progress_tracker.steps}")

            progress_bar_config = {
                "desc": "Training",
                "total": self.total_steps,
                "disable": is_progressbar_disabled(),
                "file": sys.stdout,
            }
            progress_bar = LudwigProgressBar(self.report_tqdm_to_ray, progress_bar_config, self.is_coordinator())

            while progress_tracker.steps < self.total_steps:
                # epoch init
                start_time = time.time()

                # Reset the metrics at the start of the next epoch
                self.model.reset_metrics()

                self.callback(lambda c: c.on_epoch_start(self, progress_tracker, save_path))

                should_break = self._train_loop(
                    params,
                    lgb_train,
                    eval_sets,
                    eval_names,
                    progress_tracker,
                    progress_bar,
                    save_path,
                    training_set,
                    validation_set,
                    test_set,
                    train_summary_writer,
                    validation_summary_writer,
                    test_summary_writer,
                    early_stopping_steps,
                )

                # ================ Post Training Epoch ================
                progress_tracker.epoch += 1
                self.callback(lambda c: c.on_epoch_end(self, progress_tracker, save_path))

                if self.is_coordinator():
                    # ========== Save training progress ==========
                    logger.debug(
                        f"Epoch {progress_tracker.epoch} took: "
                        f"{time_utils.strdelta((time.time()- start_time) * 1000.0)}."
                    )
                    if not self.skip_save_progress:
                        checkpoint_manager.checkpoint.model = self.model
                        checkpoint_manager.save(progress_tracker.steps)
                        progress_tracker.save(os.path.join(save_path, TRAINING_PROGRESS_TRACKER_FILE_NAME))

                # Early stop if needed.
                if should_break:
                    break
        finally:
            # ================ Finished Training ================
            if not self.model.has_saved(save_path) and self.is_coordinator() and not self.skip_save_model:
                try:
                    self.model.save(save_path)
                except ValueError:
                    logger.info("No LightGBM model initialized, skipping save")

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

            if self.is_coordinator() and not self.skip_save_progress:
                checkpoint_manager.close()

        # Load the best weights from saved checkpoint
        if self.is_coordinator() and not self.skip_save_model:
            self.model.load(save_path)

        # restore original sigint signal handler
        if self.original_sigint_handler and threading.current_thread() == threading.main_thread():
            signal.signal(signal.SIGINT, self.original_sigint_handler)

        return (
            self.model,
            progress_tracker.train_metrics,
            progress_tracker.validation_metrics,
            progress_tracker.test_metrics,
        )

    def set_steps_to_1_or_quit(self, signum, frame):
        """Custom SIGINT handler used to elegantly exit training.

        A single SIGINT will stop training after the next training step. A second SIGINT will stop training immediately.
        """
        if not self.received_sigint:
            self.total_steps = 1
            self.received_sigint = True
            logging.critical("\nReceived SIGINT, will finish this training step and then conclude training.")
            logging.critical("Send another SIGINT to immediately interrupt the process.")
        else:
            logging.critical("\nReceived a second SIGINT, will now quit")
            if self.original_sigint_handler:
                signal.signal(signal.SIGINT, self.original_sigint_handler)
            sys.exit(1)

    def _construct_lgb_params(self) -> Tuple[dict, dict]:
        output_params = {}
        feature = get_single_output_feature(self.model)

        if feature.type() == BINARY or (hasattr(feature, "num_classes") and feature.num_classes == 2):
            # To retrieve raw logit values during training, we need to use custom objective function.
            output_params = {
                "objective": log_loss_objective,
                "metric": ["binary_logloss"],
            }
        elif feature.type() == CATEGORY:
            # To retrieve raw logit values during training, we need to use custom objective function.
            output_params = {
                "objective": multiclass_objective,
                "metric": ["multi_logloss"],
                "num_class": feature.num_classes,
            }
        elif feature.type() == NUMBER:
            # Logits are not transformed by LightGBM, so no need to use custom objective function.
            output_params = {
                "objective": "regression",
                "metric": ["l2", "l1"],
            }
        else:
            raise ValueError(
                f"Model type GBM only supports numerical, categorical, or binary output features,"
                f" found: {feature.type()}"
            )

        # from: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
        params = {
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "learning_rate": self.base_learning_rate,
            "max_depth": self.max_depth,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "pos_bagging_fraction": self.pos_bagging_fraction,
            "neg_bagging_fraction": self.neg_bagging_fraction,
            "bagging_seed": self.bagging_seed,
            "bagging_freq": self.bagging_freq,
            "feature_fraction_bynode": self.feature_fraction_bynode,
            "feature_fraction_seed": self.feature_fraction_seed,
            "extra_trees": self.extra_trees,
            "extra_seed": self.extra_seed,
            "max_delta_step": self.max_delta_step,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "linear_lambda": self.linear_lambda,
            "min_gain_to_split": self.min_gain_to_split,
            "drop_rate": self.drop_rate,
            "max_drop": self.max_drop,
            "skip_drop": self.skip_drop,
            "xgboost_dart_mode": self.xgboost_dart_mode,
            "uniform_drop": self.uniform_drop,
            "drop_seed": self.drop_seed,
            "top_rate": self.top_rate,
            "other_rate": self.other_rate,
            "min_data_per_group": self.min_data_per_group,
            "max_cat_threshold": self.max_cat_threshold,
            "cat_l2": self.cat_l2,
            "cat_smooth": self.cat_smooth,
            "max_cat_to_onehot": self.max_cat_to_onehot,
            "cegb_tradeoff": self.cegb_tradeoff,
            "cegb_penalty_split": self.cegb_penalty_split,
            "path_smooth": self.path_smooth,
            "verbose": self.verbose,
            "max_bin": self.max_bin,
            "tree_learner": self.tree_learner,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_sum_hessian_in_leaf": self.min_sum_hessian_in_leaf,
            "feature_pre_filter": self.feature_pre_filter,
            "seed": self.random_seed,
            **output_params,
        }

        return params

    def _construct_lgb_datasets(
        self,
        training_set: "Dataset",  # noqa: F821
        validation_set: Optional["Dataset"] = None,  # noqa: F821
        test_set: Optional["Dataset"] = None,  # noqa: F821
    ) -> Tuple[lgb.Dataset, List[lgb.Dataset], List[str]]:
        X_train = training_set.to_df(self.model.input_features.values())
        y_train = training_set.to_df(self.model.output_features.values())

        # create dataset for lightgbm
        # keep raw data for continued training https://github.com/microsoft/LightGBM/issues/4965#issuecomment-1019344293
        try:
            lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False).construct()
        except lgb.basic.LightGBMError as e:
            if re.search(r"special JSON characters", str(e)):
                raise ValueError(
                    "Some column names in the training set contain invalid characters. Please ensure column names only "
                    "contain alphanumeric characters and underscores, then try training again."
                ) from e
            else:
                raise

        eval_sets = [lgb_train]
        eval_names = [LightGBMTrainer.TRAIN_KEY]
        if validation_set is not None:
            X_val = validation_set.to_df(self.model.input_features.values())
            y_val = validation_set.to_df(self.model.output_features.values())
            try:
                lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=False).construct()
            except lgb.basic.LightGBMError as e:
                if re.search(r"special JSON characters", str(e)):
                    raise ValueError(
                        "Some column names in the validation set contain invalid characters. Please ensure column "
                        "names only contain alphanumeric characters and underscores, then try training again."
                    ) from e
                else:
                    raise
            eval_sets.append(lgb_val)
            eval_names.append(LightGBMTrainer.VALID_KEY)
        else:
            # TODO(joppe): take X% from train set as validation set
            pass

        if test_set is not None:
            X_test = test_set.to_df(self.model.input_features.values())
            y_test = test_set.to_df(self.model.output_features.values())
            try:
                lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train, free_raw_data=False).construct()
            except lgb.basic.LightGBMError as e:
                if re.search(r"special JSON characters", str(e)):
                    raise ValueError(
                        "Some column names in the test set contain invalid characters. Please ensure column "
                        "names only contain alphanumeric characters and underscores, then try training again."
                    )
                else:
                    raise
            eval_sets.append(lgb_test)
            eval_names.append(LightGBMTrainer.TEST_KEY)

        return lgb_train, eval_sets, eval_names

    def is_coordinator(self) -> bool:
        return self.distributed.rank() == 0

    def callback(self, fn, coordinator_only=True):
        if not coordinator_only or self.is_coordinator():
            for callback in self.callbacks:
                fn(callback)


def _map_to_lgb_ray_params(params: Dict[str, Any]) -> "RayParams":  # noqa
    from lightgbm_ray import RayParams

    ray_params = {}
    for key, value in params.items():
        if key == "num_workers":
            ray_params["num_actors"] = value
        elif key == "resources_per_worker":
            if "CPU" in value:
                ray_params["cpus_per_actor"] = value["CPU"]
            if "GPU" in value:
                ray_params["gpus_per_actor"] = value["GPU"]
    ray_params = RayParams(**ray_params)
    ray_params.allow_less_than_two_cpus = True
    return ray_params


@register_ray_trainer(MODEL_GBM)
class LightGBMRayTrainer(LightGBMTrainer):
    def __init__(
        self,
        config: GBMTrainerConfig,
        model: GBM,
        resume: float = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        callbacks: List = None,
        random_seed: float = default_random_seed,
        distributed: Optional[DistributedStrategy] = None,
        device: Optional[str] = None,
        trainer_kwargs: Optional[Dict] = {},
        data_loader_kwargs: Optional[Dict] = None,
        executable_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            model=model,
            resume=resume,
            skip_save_model=skip_save_model,
            skip_save_progress=skip_save_progress,
            skip_save_log=skip_save_log,
            callbacks=callbacks,
            random_seed=random_seed,
            distributed=distributed,
            device=device,
            **kwargs,
        )

        self.ray_params = _map_to_lgb_ray_params(trainer_kwargs)
        self.data_loader_kwargs = data_loader_kwargs or {}
        self.executable_kwargs = executable_kwargs or {}

    @staticmethod
    def get_schema_cls() -> BaseTrainerConfig:
        return GBMTrainerConfig

    def train_step(
        self,
        params: Dict[str, Any],
        lgb_train: "RayDMatrix",  # noqa: F821
        eval_sets: List["RayDMatrix"],  # noqa: F821
        eval_names: List[str],
        init_model: lgb.LGBMModel,
        boost_rounds_per_train_step: int,
        evals_result: Dict,
    ) -> lgb.LGBMModel:
        """Trains a LightGBM model using ray.

        Args:
            params: parameters for LightGBM
            lgb_train: RayDMatrix dataset for training
            eval_sets: RayDMatrix datasets for evaluation
            eval_names: names of the evaluation datasets

        Returns:
            LightGBM Booster model
        """
        from lightgbm_ray import RayLGBMClassifier, RayLGBMRegressor

        output_feature = get_single_output_feature(self.model)
        gbm_sklearn_cls = RayLGBMRegressor if output_feature.type() == NUMBER else RayLGBMClassifier

        additional_results = {}
        gbm = gbm_sklearn_cls(n_estimators=boost_rounds_per_train_step, **params).fit(
            X=lgb_train,
            y=None,
            init_model=init_model,
            eval_set=[(s, n) for s, n in zip(eval_sets, eval_names)],
            eval_names=eval_names,
            callbacks=[
                # add early stopping callback to populate best_iteration
                lgb.early_stopping(boost_rounds_per_train_step),
                store_predictions_ray(boost_rounds_per_train_step),
            ],
            additional_results=additional_results,
            ray_params=self.ray_params,
        )
        evals_result.update(gbm.evals_result_)

        if not self.evaluate_training_set:
            # Update evaluation metrics with current model params:
            # noisy but fast way to get metrics on the training set

            # Only use the first actor's predictions.
            actor_callback_return = next(iter(additional_results["callback_returns"]))
            # Only a single TrainLogits is expected (final iteration).
            train_logits: TrainLogits = next(iter(actor_callback_return))
            predictions = logits_to_predictions(self.model, train_logits.preds)

            targets = get_targets(lgb_train, output_feature, self.device, actor_rank=0)

            self.model.update_metrics(targets, predictions)

        return gbm.to_local()

    def _construct_lgb_datasets(
        self,
        training_set: "RayDataset",  # noqa: F821
        validation_set: Optional["RayDataset"] = None,  # noqa: F821
        test_set: Optional["RayDataset"] = None,  # noqa: F821
    ) -> Tuple["RayDMatrix", List["RayDMatrix"], List[str]]:  # noqa: F821
        """Prepares Ludwig RayDataset objects for use in LightGBM."""

        from lightgbm_ray import RayDMatrix

        output_feature = get_single_output_feature(self.model)
        label_col = output_feature.proc_column

        in_feat = [f.proc_column for f in self.model.input_features.values()]
        out_feat = [f.proc_column for f in self.model.output_features.values()]
        feat_cols = in_feat + out_feat

        # TODO(shreya): Refactor preprocessing so that this can be moved upstream.
        if training_set.ds.num_blocks() < self.ray_params.num_actors:
            # Repartition to ensure that there is at least one block per actor
            training_set.repartition(self.ray_params.num_actors)

        lgb_train = RayDMatrix(
            # NOTE: batch_size=None to make sure map_batches doesn't change num_blocks.
            # Need num_blocks to equal num_actors in order to feed all actors.
            training_set.ds.map_batches(lambda df: df[feat_cols], batch_size=None),
            label=label_col,
        )

        eval_sets = [lgb_train]
        eval_names = [LightGBMTrainer.TRAIN_KEY]
        if validation_set is not None:
            if validation_set.ds.num_blocks() < self.ray_params.num_actors:
                # Repartition to ensure that there is at least one block per actor
                validation_set.repartition(self.ray_params.num_actors)

            lgb_val = RayDMatrix(
                validation_set.ds.map_batches(lambda df: df[feat_cols], batch_size=None),
                label=label_col,
            )
            eval_sets.append(lgb_val)
            eval_names.append(LightGBMTrainer.VALID_KEY)

        if test_set is not None:
            if test_set.ds.num_blocks() < self.ray_params.num_actors:
                # Repartition to ensure that there is at least one block per actor
                test_set.repartition(self.ray_params.num_actors)

            lgb_test = RayDMatrix(
                test_set.ds.map_batches(lambda df: df[feat_cols], batch_size=None),
                label=label_col,
            )
            eval_sets.append(lgb_test)
            eval_names.append(LightGBMTrainer.TEST_KEY)

        return lgb_train, eval_sets, eval_names
