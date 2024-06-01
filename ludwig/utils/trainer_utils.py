import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUTO, COMBINED, LOSS
from ludwig.models.base import BaseModel
from ludwig.models.ecd import ECD
from ludwig.models.llm import LLM
from ludwig.modules.metric_modules import get_best_function
from ludwig.schema.trainer import ECDTrainerConfig, FineTuneTrainerConfig
from ludwig.utils.data_utils import save_json
from ludwig.utils.metric_utils import TrainerMetric

if TYPE_CHECKING:
    from ludwig.features.base_feature import OutputFeature
    from ludwig.schema.trainer import BaseTrainerConfig


logger = logging.getLogger(__name__)


@DeveloperAPI
def initialize_trainer_metric_dict(output_features) -> Dict[str, Dict[str, List[TrainerMetric]]]:
    """Returns a dict of dict of metrics, output_feature_name -> metric_name -> List[TrainerMetric]."""
    metrics = defaultdict(lambda: defaultdict(list))
    return metrics


def get_latest_metrics_dict(
    progress_tracker_metrics: Dict[str, Dict[str, List[TrainerMetric]]]
) -> Dict[str, Dict[str, float]]:
    """Returns a dict of field name -> metric name -> latest metric value."""
    latest_metrics_dict = defaultdict(dict)
    for feature_name, metrics_dict in progress_tracker_metrics.items():
        for metric_name, metrics in metrics_dict.items():
            if metrics:
                # Metrics may be missing if computing metrics was excepted, if the metrics are entirely empty
                # due to a missing subset, or if evaluate_training_set is False.
                latest_metrics_dict[feature_name][metric_name] = metrics[-1][-1]
    return latest_metrics_dict


@DeveloperAPI
def get_new_progress_tracker(
    batch_size: int,
    best_eval_metric_value: float,
    best_increase_batch_size_eval_metric: float,
    learning_rate: float,
    output_features: Dict[str, "OutputFeature"],
):
    """Returns a new instance of a ProgressTracker with empty metrics."""
    return ProgressTracker(
        epoch=0,
        batch_size=batch_size,
        steps=0,
        tune_checkpoint_num=0,
        checkpoint_number=0,
        best_eval_metric_steps=0,
        best_eval_metric_epoch=0,
        best_eval_metric_checkpoint_number=0,
        last_learning_rate_reduction_steps=0,
        last_increase_batch_size_steps=0,
        last_improvement_steps=0,
        best_eval_metric_value=best_eval_metric_value,
        best_increase_batch_size_eval_metric=best_increase_batch_size_eval_metric,
        last_increase_batch_size_eval_metric_improvement=0,
        learning_rate=learning_rate,
        num_reductions_learning_rate=0,
        num_increases_batch_size=0,
        train_metrics=initialize_trainer_metric_dict(output_features),
        validation_metrics=initialize_trainer_metric_dict(output_features),
        test_metrics=initialize_trainer_metric_dict(output_features),
        last_learning_rate_reduction=0,
        last_increase_batch_size=0,
        best_eval_train_metrics={},
        best_eval_validation_metrics={},
        best_eval_test_metrics={},
        llm_eval_examples={},
        checkpoint_to_step={},
        checkpoint_to_epoch={},
        incremental_step_token_usage={},
        cumulative_step_token_usage={},
        incremental_checkpoint_token_usage={},
        cumulative_checkpoint_token_usage={},
        total_tokens_used=0,
    )


@DeveloperAPI
class ProgressTracker:
    def __init__(
        self,
        epoch: int,
        batch_size: int,
        steps: int,
        tune_checkpoint_num: int,
        checkpoint_number: int,
        best_eval_metric_steps: int,
        best_eval_metric_epoch: int,
        best_eval_metric_checkpoint_number: int,
        last_improvement_steps: int,
        last_learning_rate_reduction_steps: int,
        last_increase_batch_size_steps: int,
        best_eval_metric_value: float,
        best_increase_batch_size_eval_metric: float,
        last_increase_batch_size_eval_metric_improvement: int,
        learning_rate: float,
        num_reductions_learning_rate: int,
        num_increases_batch_size: int,
        train_metrics: Dict[str, Dict[str, List[TrainerMetric]]],
        validation_metrics: Dict[str, Dict[str, List[TrainerMetric]]],
        test_metrics: Dict[str, Dict[str, List[TrainerMetric]]],
        last_learning_rate_reduction: int,
        last_increase_batch_size: int,
        best_eval_train_metrics: Dict[str, Dict[str, float]],
        best_eval_validation_metrics: Dict[str, Dict[str, float]],
        best_eval_test_metrics: Dict[str, Dict[str, float]],
        llm_eval_examples: Dict[str, List[str]] = None,
        checkpoint_to_step: Dict[str, int] = None,
        checkpoint_to_epoch: Dict[str, int] = None,
        incremental_step_token_usage: Dict[str, int] = None,
        cumulative_step_token_usage: Dict[str, int] = None,
        incremental_checkpoint_token_usage: Dict[str, int] = None,
        cumulative_checkpoint_token_usage: Dict[str, int] = None,
        total_tokens_used: int = 0,
    ):
        """JSON-serializable holder object that stores information related to training progress.

        [train/vali/test]_metrics is a nested dictionary of TrainerMetrics: feature_name -> metric_name ->
        List[TrainerMetrics], with one entry per training checkpoint.

        When the model is saved, all of the progress tracker's attributes are serialized to JSON as
        `training_progress.json` under the model output directory.

        JSON serialization automatically converts all dictionary top-level keys to strings, and the string typing
        is preserved when the progress tracker is deserialized from JSON when model resumes training from a checkpoint.

        For this reason, all of the dictionary attributes of the progress tracker are keyed by strings to ensure a
        consistent interface before or after deserialization. For example, the `tokens` dictionaries are keyed by steps,
        as strings.

        When the progress tracker is deserialized from JSON like when a model resumes training from a checkpoint, the
        TrainerMetrics namedtuples are automatically converted into regular (epoch, steps, value) tuples, which is why
        in trainer.py, we often use `[-1]` to index into the last element of the TrainerMetric namedtuple to get the
        actual metric value instead of the named field.

        Args:
            epoch: The current epoch number.
            steps: The current step of training.
            batch_size: The current batch size.
            tune_checkpoint_num: The hyperopt checkpoint number (Ray Tune).
            checkpoint_number: The current checkpoint number.

            best_eval_metric_steps: The step of training that has the best evaluation so far.
            best_eval_metric_epoch: The epoch of training that has the best evaluation so far.
            best_eval_metric_checkpoint_number: The checkpoint number that has the best evaluation so far.

            last_improvement_steps: The number of steps since the last improvement.
            last_learning_rate_reduction_steps: The training step of the last learning rate reduction.
            last_increase_batch_size_steps: The training_step of the the last batch size increase.

            best_eval_metric_value: The metric value of the best evaluation so far.
            best_increase_batch_size_eval_metric:
                The metric value of the best evaluation so far, for increasing the batch size.

            last_learning_rate_reduction: The number of steps since the last learning rate reduction.
            last_increase_batch_size: The number of steps since the last batch size increase.

            last_increase_batch_size_eval_metric_improvement:
                The number of checkpoints since the last batch size increase.

            num_reductions_learning_rate: The number of total reductions in learning rate.
            num_increases_batch_size: The number of total increases in batch size.

            train_metrics: Training metrics. <output feature name> -> <metric name> -> History of metrics.
            validation_metrics: Validation metrics. <output feature name> -> <metric name> -> History of metrics.
            test_metrics: Test metrics. <output feature name> -> <metric name> -> History of metrics.

            best_eval_train_metrics:
                Best eval train metrics: <output feature name> -> <metric name> -> <metric value>.
            best_eval_validation_metrics:
                Best eval validation metrics: <output feature name> -> <metric name> -> <metric value>.
            best_eval_test_metrics:
                Best eval test metrics: <output feature name> -> <metric name> -> <metric value>.

            llm_eval_examples:
                Dictionary whose keys are "inputs", "targets", and "outputs" and whose values are dicts.
                The keys of each subdict are the names of the input/target/output features and the values are lists of
                example tensors. This is only set for LLM fine-tuning.

            checkpoint_to_step: Map of checkpoint number to step number.
            checkpoint_to_epoch: Map of checkpoint number to epoch number.

            incremental_step_token_usage: Map of step number to number of tokens used in that step.
            cumulative_step_token_usage: Map of step number to cumulative number of tokens used up to that step.
            incremental_checkpoint_token_usage: Map of checkpoint number to number of tokens used up to that checkpoint
                since the last checkpoint.
            cumulative_checkpoint_token_usage: Map of checkpoint number to cumulative number of tokens used up to that
                checkpoint.
            total_tokens_used: Total number of tokens used.
        """
        self.batch_size = batch_size
        self.epoch = epoch
        self.steps = steps
        self.tune_checkpoint_num = tune_checkpoint_num
        self.checkpoint_number = checkpoint_number
        self.best_eval_metric_steps = best_eval_metric_steps
        self.best_eval_metric_epoch = best_eval_metric_epoch
        self.best_eval_metric_checkpoint_number = best_eval_metric_checkpoint_number
        self.last_improvement_steps = last_improvement_steps
        self.last_learning_rate_reduction_steps = last_learning_rate_reduction_steps
        self.last_learning_rate_reduction = last_learning_rate_reduction
        self.last_increase_batch_size_steps = last_increase_batch_size_steps
        self.last_increase_batch_size = last_increase_batch_size
        self.learning_rate = learning_rate
        self.best_eval_metric_value = best_eval_metric_value
        self.best_increase_batch_size_eval_metric = best_increase_batch_size_eval_metric
        self.last_increase_batch_size_eval_metric_improvement = last_increase_batch_size_eval_metric_improvement
        self.num_reductions_learning_rate = num_reductions_learning_rate
        self.num_increases_batch_size = num_increases_batch_size
        self.train_metrics = train_metrics
        self.validation_metrics = validation_metrics
        self.test_metrics = test_metrics

        # This should be an dictionary whose keys are "inputs", "targets", and "outputs" and whose values are dicts.
        # The keys of each subdict are the names of the input/target/output features and the values are lists of
        # example tensors. This is only set for LLM fine-tuning.
        self.llm_eval_examples = llm_eval_examples

        # Best metrics.
        self.best_eval_train_metrics = best_eval_train_metrics
        self.best_eval_validation_metrics = best_eval_validation_metrics
        self.best_eval_test_metrics = best_eval_test_metrics

        # Checkpoint tracking.
        self.checkpoint_to_step = checkpoint_to_step
        self.checkpoint_to_epoch = checkpoint_to_epoch

        # Token usage.
        self.incremental_step_token_usage = incremental_step_token_usage
        self.cumulative_step_token_usage = cumulative_step_token_usage
        self.incremental_checkpoint_token_usage = incremental_checkpoint_token_usage
        self.cumulative_checkpoint_token_usage = cumulative_checkpoint_token_usage
        self.total_tokens_used = total_tokens_used

    def save(self, filepath):
        # sort_keys=False to ensure that token usage dictionaries (keyed by integers) are encodable.
        # save_json(filepath, self.__dict__, sort_keys=False)
        save_json(filepath, self.__dict__)

    @staticmethod
    def load(progress_tracking_dict: Dict):
        from ludwig.utils.backward_compatibility import upgrade_model_progress

        loaded = upgrade_model_progress(progress_tracking_dict)
        return ProgressTracker(**loaded)

    def log_metrics(self):
        log_metrics = {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "steps": self.steps,
            "tune_checkpoint_num": self.tune_checkpoint_num,
            "checkpoint_number": self.checkpoint_number,
            "last_improvement_steps": self.last_improvement_steps,
            "best_eval_metric_steps": self.best_eval_metric_steps,
            "best_eval_metric_epoch": self.best_eval_metric_epoch,
            "best_eval_metric_checkpoint_number": self.best_eval_metric_checkpoint_number,
            "learning_rate": self.learning_rate,
            "best_valid_metric": self.best_eval_metric_value,
            "num_reductions_lr": self.num_reductions_learning_rate,
            "num_increases_bs": self.num_increases_batch_size,
            "total_tokens_used": self.total_tokens_used,
        }

        # This is a non-numerical metric that is only for LLM fine-tuning
        # This should be an dictionary whose keys are "inputs", "targets", and "outputs" and whose values are dicts.
        # The keys of each subdict are the names of the input/target/output features and the values are lists of
        # example tensors.
        if self.llm_eval_examples:
            log_metrics["llm_eval_examples"] = self.llm_eval_examples

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
                        #
                        # TODO: when loading an existing model, this loses metric values for all but the last epoch.
                        log_metrics[f"{metrics_dict_name}.{feature_name}.{metric_name}"] = metrics_tuples[-1][-1]

        # Add best metrics.
        for feature_name, metrics in self.best_eval_train_metrics.items():
            for metric_name, metric_value in metrics.items():
                log_metrics[f"best.train_metrics.{feature_name}.{metric_name}"] = metric_value
        for feature_name, metrics in self.best_eval_validation_metrics.items():
            for metric_name, metric_value in metrics.items():
                log_metrics[f"best.validation_metrics.{feature_name}.{metric_name}"] = metric_value
        for feature_name, metrics in self.best_eval_test_metrics.items():
            for metric_name, metric_value in metrics.items():
                log_metrics[f"best.test_metrics.{feature_name}.{metric_name}"] = metric_value

        return log_metrics

    def _add_checkpoint_entry_for_used_tokens(self, checkpoint_number: int):
        """Adds an entry to the token usage dictionaries for the given checkpoint number.

        Assumes that the token usage dictionaries for steps are filled.
        """
        self.cumulative_checkpoint_token_usage[str(checkpoint_number)] = self.total_tokens_used

        if checkpoint_number <= 0:
            raise ValueError("Checkpoint number should be greater than 0.")

        if checkpoint_number == 1:
            # The incremental token usage for checkpoint 0 is the same as the total tokens used so far.
            self.incremental_checkpoint_token_usage[str(checkpoint_number)] = self.total_tokens_used
        else:
            # The incremental token usage for this checkpoint is the total tokens used minus the cumulative tokens used
            # up to the previous checkpoint.
            previous_checkpoint_number = checkpoint_number - 1

            tokens_used_since_previous_checkpoint = (
                self.total_tokens_used - self.cumulative_checkpoint_token_usage[str(previous_checkpoint_number)]
            )
            self.incremental_checkpoint_token_usage[str(checkpoint_number)] = tokens_used_since_previous_checkpoint

    def increment_checkpoint(self):
        """Update the progress tracker for a new checkpoint."""
        self.checkpoint_number += 1

        # Set checkpoint -> step/epoch lookup maps.
        self.checkpoint_to_step[str(self.checkpoint_number)] = self.steps
        self.checkpoint_to_epoch[str(self.checkpoint_number)] = self.epoch

        # Set checkpoint -> used tokens lookup maps.
        self._add_checkpoint_entry_for_used_tokens(self.checkpoint_number)

    def set_token_usage_for_this_step(self, used_tokens: int):
        """Update the token usage for the current step."""
        steps_str = str(self.steps)
        self.incremental_step_token_usage[steps_str] = used_tokens
        self.total_tokens_used += used_tokens
        self.cumulative_step_token_usage[steps_str] = self.total_tokens_used


@DeveloperAPI
def append_metrics(
    model: BaseModel,
    dataset_name: Literal["train", "validation", "test"],
    results: Dict[str, Dict[str, float]],
    metrics_log: Dict[str, Dict[str, List[TrainerMetric]]],
    progress_tracker: ProgressTracker,
) -> Dict[str, Dict[str, List[TrainerMetric]]]:
    epoch = progress_tracker.epoch
    steps = progress_tracker.steps
    for output_feature in model.output_features:
        scores = [dataset_name]

        # collect metric names based on output features metrics to
        # ensure consistent order of reporting metrics
        metric_names = sorted(results[output_feature].keys())

        for metric in metric_names:
            if metric in results[output_feature]:
                # Some metrics may have been excepted and excluded from results.
                score = results[output_feature][metric]
                metrics_log[output_feature][metric].append(TrainerMetric(epoch=epoch, step=steps, value=score))
                scores.append(score)

    metrics_log[COMBINED][LOSS].append(TrainerMetric(epoch=epoch, step=steps, value=results[COMBINED][LOSS]))
    return metrics_log


@DeveloperAPI
def get_total_steps(epochs: int, steps_per_epoch: int, train_steps: int):
    """Returns train_steps if provided, otherwise epochs * steps_per_epoch."""
    if train_steps:
        return train_steps
    return epochs * steps_per_epoch


@DeveloperAPI
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
            logger.info(
                f"Note: steps_per_checkpoint (was {steps_per_checkpoint}) is now set to the number of "
                f"steps per epoch: {steps_per_epoch}.\n"
            )
        return steps_per_epoch

    # steps_per_checkpoint wasn't specified. Use steps_per_epoch.
    if steps_per_checkpoint == 0:
        return steps_per_epoch

    return steps_per_checkpoint


def get_total_expected_checkpoints(total_steps: int, final_steps_per_checkpoint: int, epochs: int) -> int:
    return total_steps // final_steps_per_checkpoint + epochs


@DeveloperAPI
def get_training_report(
    validation_field: str,
    validation_metric: str,
    include_test_set: bool,
    train_valiset_stats: Dict[str, Dict[str, List[float]]],
    train_testset_stats: Dict[str, Dict[str, List[float]]],
) -> List[Tuple[str, str]]:
    """Returns a training report in the form of a list [(report item, value)]."""
    validation_field_result = train_valiset_stats[validation_field]
    best_function = get_best_function(validation_metric)

    training_report = []
    best_vali_index, (
        epoch_best_validation_metric,
        step_best_validation_metric,
        best_validation_metric,
    ) = best_function(
        enumerate(validation_field_result[validation_metric]),
        # -1 for the last element of the TrainerMetric namedtuple.
        key=lambda index_epoch_step_value: index_epoch_step_value[1][-1],
    )
    training_report.append(["Validation feature", validation_field])
    training_report.append(["Validation metric", validation_metric])
    training_report.append(["Best model step", step_best_validation_metric])
    training_report.append(["Best model epoch", epoch_best_validation_metric + 1])
    training_report.append(
        [
            f"Best model's validation {validation_metric}",
            best_validation_metric,
        ]
    )
    if include_test_set:
        validation_selected_test_metric_score = train_testset_stats[validation_field][validation_metric][
            best_vali_index
        ][
            -1
        ]  # -1 for the last element of the TrainerMetric namedtuple.

        training_report.append(
            [
                f"Best model's test {validation_metric}",
                validation_selected_test_metric_score,
            ]
        )
    return training_report


def get_rendered_batch_size_grad_accum(config: "BaseTrainerConfig", num_workers: int) -> Tuple[int, int]:
    """Returns the batch size and gradient accumulation steps to use for training.

    For batch_size==AUTO:
    1. effective_batch_size is not AUTO and gradient_accumulation_steps is not AUTO:
        batch size is set to the effective batch size divided by the gradient accumulation steps, divided by the
        number of workers.
    2. effective_batch_size is AUTO or gradient_accumulation_steps is AUTO:
        batch size remains AUTO.

    For gradient_accumulation_steps==AUTO:
    1. batch size is AUTO:
        gradient accumulation steps remains AUTO.
    2. batch_size is not AUTO and effective batch size is not AUTO:
        gradient accumulation steps is set to the effective batch size divided by the batch size, divided by the number
        of workers.
    3. batch size is not AUTO and effective batch size is AUTO:
        gradient accumulation steps is set to 1.
    """
    effective_batch_size = config.effective_batch_size
    batch_size = config.batch_size
    gradient_accumulation_steps = config.gradient_accumulation_steps

    if config.batch_size == AUTO:
        if config.effective_batch_size != AUTO and config.gradient_accumulation_steps != AUTO:
            batch_size = max(int(effective_batch_size / gradient_accumulation_steps / num_workers), 1)

    if config.gradient_accumulation_steps == AUTO:
        if config.batch_size != AUTO:
            if config.effective_batch_size != AUTO:
                gradient_accumulation_steps = max(int(effective_batch_size / batch_size / num_workers), 1)
            else:
                gradient_accumulation_steps = 1

    return batch_size, gradient_accumulation_steps


def freeze_layers_regex(config: Union[ECDTrainerConfig, FineTuneTrainerConfig], model: Union[ECD, LLM]) -> None:
    """Freezes layers in a model whose names match a specified regular expression pattern.

    This function iterates over all parameters of the model, checking each parameter's name against
    the regular expression defined in the configuration object.
    If a match is found, the parameter's `requires_grad` attribute is set to False,
    effectively freezing the layer for training purposes.
    If no matches are found, an error is logged indicating the issue with the regex or the model's layer names.

    Parameters:
    - config (Union[ECDTrainerConfig, FineTuneTrainerConfig]):
    - model (Union[ECD, LLM]): The model object containing layers and parameters. This could be an instance of either
    ECD or LLM classes, which should have a method `named_parameters()` that yields the name and parameter
    object of each layer.

    Raises:
    - re.error: If the regular expression pattern in `config.layers_to_freeze_regex` is invalid, an error is logged
    and the function exits.

    Returns:
    - None: This function does not return any value but modifies the model in-place by freezing certain layers.
    """
    pattern = re.compile(config.layers_to_freeze_regex)
    matched_layers = set()

    for name, p in model.named_parameters():
        if re.search(pattern, str(name)):
            p.requires_grad = False
            matched_layers.add(name)
    if matched_layers:
        logger.info(f"Layers where requires_grad was set to False: {matched_layers}")
    else:
        logger.error(f"No regex match for {config.layers_to_freeze_regex}! Check layer names and regex syntax.")

    count_parameters(model)


def count_parameters(model) -> None:
    """Counts number of trainable parameters post freezing.

    Returns:
    - None: This function does not return any value.
    """
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()

        total_params += params

    logger.info(f"Total Trainable Parameters after freezing: {total_params}")
