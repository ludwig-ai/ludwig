import logging
from collections import OrderedDict
from typing import Dict, List

from ludwig.constants import COMBINED, LOSS
from ludwig.features.base_feature import OutputFeature
from ludwig.utils.data_utils import load_json, save_json
from ludwig.utils.metric_utils import TrainerMetric


def initialize_trainer_metric_dict(output_features) -> Dict[str, Dict[str, List[TrainerMetric]]]:
    """Returns a dict of dict of metrics, output_feature_name -> metric_name -> List[TrainerMetric]."""
    metrics = OrderedDict()

    for output_feature_name, output_feature in output_features.items():
        metrics[output_feature_name] = OrderedDict()
        for metric in output_feature.metric_functions:
            metrics[output_feature_name][metric] = []

    metrics[COMBINED] = {LOSS: []}
    return metrics


def get_new_progress_tracker(
    batch_size: int,
    best_eval_metric: float,
    best_reduce_learning_rate_eval_metric: float,
    best_increase_batch_size_eval_metric: float,
    learning_rate: float,
    output_features: Dict[str, OutputFeature],
):
    """Returns a new instance of a ProgressTracker with empty metrics."""
    return ProgressTracker(
        epoch=0,
        batch_size=batch_size,
        steps=0,
        tune_checkpoint_num=0,
        last_improvement_steps=0,
        last_learning_rate_reduction_steps=0,
        last_increase_batch_size_steps=0,
        best_eval_metric=best_eval_metric,
        best_reduce_learning_rate_eval_metric=best_reduce_learning_rate_eval_metric,
        last_reduce_learning_rate_eval_metric_improvement=0,
        best_increase_batch_size_eval_metric=best_increase_batch_size_eval_metric,
        last_increase_batch_size_eval_metric_improvement=0,
        learning_rate=learning_rate,
        num_reductions_learning_rate=0,
        num_increases_batch_size=0,
        train_metrics=initialize_trainer_metric_dict(output_features),
        validation_metrics=initialize_trainer_metric_dict(output_features),
        test_metrics=initialize_trainer_metric_dict(output_features),
        last_improvement=0,
        last_learning_rate_reduction=0,
        last_increase_batch_size=0,
    )


class ProgressTracker:
    def __init__(
        self,
        epoch: int,
        batch_size: int,
        steps: int,
        tune_checkpoint_num: int,
        last_improvement_steps: int,
        last_learning_rate_reduction_steps: int,
        last_increase_batch_size_steps: int,
        best_eval_metric: float,
        best_reduce_learning_rate_eval_metric: float,
        last_reduce_learning_rate_eval_metric_improvement: int,
        best_increase_batch_size_eval_metric: float,
        last_increase_batch_size_eval_metric_improvement: int,
        learning_rate: float,
        num_reductions_learning_rate: int,
        num_increases_batch_size: int,
        train_metrics: Dict[str, Dict[str, List[TrainerMetric]]],
        validation_metrics: Dict[str, Dict[str, List[TrainerMetric]]],
        test_metrics: Dict[str, Dict[str, List[TrainerMetric]]],
        last_improvement: int,
        last_learning_rate_reduction: int,
        last_increase_batch_size: int,
    ):
        """JSON-serializable holder object that stores information related to training progress.

        [train/vali/test]_metrics is a nested dictionary of TrainerMetrics: feature_name -> metric_name ->
        List[TrainerMetrics], with one entry per training checkpoint.

        Note that when a model resumes training from a checkpoint, the progress tracker is deserialized from JSON, which
        automatically converts TrainerMetrics namedtuples into regular (epoch, steps, value) tuples.
        """
        self.batch_size = batch_size
        self.epoch = epoch
        self.steps = steps
        self.tune_checkpoint_num = tune_checkpoint_num
        self.last_improvement_steps = last_improvement_steps
        self.last_improvement = last_improvement
        self.last_learning_rate_reduction_steps = last_learning_rate_reduction_steps
        self.last_learning_rate_reduction = last_learning_rate_reduction
        self.last_increase_batch_size_steps = last_increase_batch_size_steps
        self.last_increase_batch_size = last_increase_batch_size
        self.learning_rate = learning_rate
        self.best_eval_metric = best_eval_metric
        self.best_reduce_learning_rate_eval_metric = best_reduce_learning_rate_eval_metric
        self.last_reduce_learning_rate_eval_metric_improvement = last_reduce_learning_rate_eval_metric_improvement
        self.best_increase_batch_size_eval_metric = best_increase_batch_size_eval_metric
        self.last_increase_batch_size_eval_metric_improvement = last_increase_batch_size_eval_metric_improvement
        self.num_reductions_learning_rate = num_reductions_learning_rate
        self.num_increases_batch_size = num_increases_batch_size
        self.train_metrics = train_metrics
        self.validation_metrics = validation_metrics
        self.test_metrics = test_metrics

    def save(self, filepath):
        save_json(filepath, self.__dict__)

    @staticmethod
    def load(filepath):
        loaded = load_json(filepath)
        return ProgressTracker(**loaded)

    def log_metrics(self):
        log_metrics = {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "steps": self.steps,
            "tune_checkpoint_num": self.tune_checkpoint_num,
            "last_improvement_steps": self.last_improvement_steps,
            "learning_rate": self.learning_rate,
            "best_valid_metric": self.best_eval_metric,
            "num_reductions_lr": self.num_reductions_learning_rate,
            "num_increases_bs": self.num_increases_batch_size,
        }
        for metrics_dict_name in [
            "train_metrics",
            "validation_metrics",
            "test_metrics",
        ]:
            metrics_dict = getattr(self, metrics_dict_name)
            for feature_name in metrics_dict:
                for metric_name, metrics_tuples in metrics_dict[feature_name].items():
                    if metrics_tuples:
                        # For logging, get the latest metrics. The second "-1" indexes into the TrainerMetric
                        # namedtuple. The last element of the TrainerMetric namedtuple is the actual metric value.
                        log_metrics[f"{metrics_dict_name}.{feature_name}.{metric_name}"] = metrics_tuples[-1][-1]

        return log_metrics


def get_total_steps(epochs: int, steps_per_epoch: int, train_steps: int):
    """Returns train_steps if non-negative.

    Otherwise, returns the number of epochs.
    """
    if train_steps:
        return train_steps
    return epochs * steps_per_epoch


def get_final_steps_per_checkpoint(
    steps_per_epoch: int, steps_per_checkpoint: int = 0, checkpoints_per_epoch: float = 0, should_log: bool = False
):
    """Returns the steps per checkpoint to use for the training loop, given user+default inputs."""
    if steps_per_checkpoint != 0 and checkpoints_per_epoch != 0:
        raise ValueError(
            "It is invalid to specify both checkpoints_per_epoch AND steps_per_checkpoint. Please specify one or the "
            "other, or specify neither to checkpoint/eval the model every epoch."
        )

    # Set steps_per_checkpoint based on the checkpoints_per_epoch, if checkpoints_per_epoch was specified.
    if checkpoints_per_epoch != 0:
        steps_per_checkpoint = int(steps_per_epoch / checkpoints_per_epoch)

    # Cap steps_per_checkpoint at steps_per_epoch.
    if steps_per_checkpoint > steps_per_epoch:
        if should_log:
            logging.info(
                f"Note: steps_per_checkpoint (was {steps_per_checkpoint}) is now set to the number of "
                f"steps per epoch: {steps_per_epoch}.\n"
            )
        return steps_per_epoch

    # steps_per_checkpoint wasn't specified. Use steps_per_epoch.
    if steps_per_checkpoint == 0:
        return steps_per_epoch

    return steps_per_checkpoint
