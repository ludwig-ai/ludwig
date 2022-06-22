from typing import Optional, Union

from marshmallow_dataclass import dataclass

import ludwig.schema.metadata.trainer as lsmt
from ludwig.constants import COMBINED, LOSS, TRAINING
from ludwig.schema import utils as schema_utils
from ludwig.schema.optimizers import (
    BaseOptimizerConfig,
    GradientClippingConfig,
    GradientClippingDataclassField,
    OptimizerDataclassField,
)


@dataclass
class TrainerConfig(schema_utils.BaseMarshmallowConfig):
    """TrainerConfig is a dataclass that configures most of the hyperparameters used for model training."""

    optimizer: BaseOptimizerConfig = OptimizerDataclassField(default={"type": "adam"}, **lsmt.optimizer_metadata)

    epochs: int = schema_utils.PositiveInteger(default=100, **lsmt.epochs_metadata)

    train_steps: int = schema_utils.PositiveInteger(default=None, **lsmt.train_steps_metadata)

    regularization_lambda: float = schema_utils.FloatRange(default=0.0, min=0, **lsmt.regularization_lambda_metadata)

    regularization_type: Optional[str] = schema_utils.RegularizerOptions(
        default="l2", **lsmt.regularization_type_metadata
    )

    should_shuffle: bool = schema_utils.Boolean(default=True, **lsmt.should_shuffle_metadata)

    learning_rate: float = schema_utils.FloatOrAutoField(
        default=0.001, min=0.0, max=1.0, default_numeric=0.001, **lsmt.learning_rate_metadata
    )

    batch_size: Union[int, str] = schema_utils.IntegerOrAutoField(
        default=128, default_numeric=128, min_exclusive=0, **lsmt.batch_size_metadata
    )

    eval_batch_size: Union[None, int, str] = schema_utils.IntegerOrAutoField(
        default=None, default_numeric=None, min_exclusive=0, **lsmt.eval_batch_size_metadata
    )

    early_stop: int = schema_utils.IntegerRange(default=5, min=-1, **lsmt.early_stop_metadata)

    steps_per_checkpoint: int = schema_utils.NonNegativeInteger(default=0, **lsmt.steps_per_checkpoint_metadata)

    checkpoints_per_epoch: int = schema_utils.NonNegativeInteger(default=0, **lsmt.checkpoints_per_epoch_metadata)

    evaluate_training_set: bool = schema_utils.Boolean(default=True, **lsmt.evaluate_training_set_metadata)

    reduce_learning_rate_on_plateau: float = schema_utils.FloatRange(
        default=0.0, min=0.0, max=1.0, **lsmt.reduce_learning_rate_on_plateau_metadata
    )

    reduce_learning_rate_on_plateau_patience: int = schema_utils.NonNegativeInteger(
        default=5, **lsmt.reduce_learning_rate_on_plateau_patience_metadata
    )

    reduce_learning_rate_on_plateau_rate: float = schema_utils.FloatRange(
        default=0.5, min=0.0, max=1.0, **lsmt.reduce_learning_rate_on_plateau_rate_metadata
    )

    reduce_learning_rate_eval_metric: str = schema_utils.String(
        default=LOSS, **lsmt.reduce_learning_rate_eval_metric_metadata
    )

    reduce_learning_rate_eval_split: str = schema_utils.String(
        default=TRAINING, **lsmt.reduce_learning_rate_eval_split_metadata
    )

    increase_batch_size_on_plateau: int = schema_utils.NonNegativeInteger(
        default=0, **lsmt.increase_batch_size_on_plateau_metadata
    )

    increase_batch_size_on_plateau_patience: int = schema_utils.NonNegativeInteger(
        default=5, **lsmt.increase_batch_size_on_plateau_patience_metadata
    )

    increase_batch_size_on_plateau_rate: float = schema_utils.NonNegativeFloat(
        default=2.0, **lsmt.increase_batch_size_on_plateau_rate_metadata
    )

    increase_batch_size_on_plateau_max: int = schema_utils.PositiveInteger(
        default=512, **lsmt.increase_batch_size_on_plateau_max_metadata
    )

    increase_batch_size_eval_metric: str = schema_utils.String(
        default=LOSS, **lsmt.increase_batch_size_eval_metric_metadata
    )

    increase_batch_size_eval_split: str = schema_utils.String(
        default=TRAINING, **lsmt.increase_batch_size_eval_split_metadata
    )

    decay: bool = schema_utils.Boolean(default=False, **lsmt.decay_metadata)

    decay_steps: int = schema_utils.PositiveInteger(default=10000, **lsmt.decay_steps_metadata)

    decay_rate: float = schema_utils.FloatRange(default=0.96, min=0.0, max=1.0, **lsmt.decay_rate_metadata)

    staircase: bool = schema_utils.Boolean(default=False, **lsmt.staircase_metadata)

    gradient_clipping: Optional[GradientClippingConfig] = GradientClippingDataclassField(**lsmt.staircase_metadata)

    # TODO(#1673): Need some more logic here for validating against output features
    validation_field: str = schema_utils.String(default=COMBINED, **lsmt.validation_field_metadata)

    validation_metric: str = schema_utils.String(default=LOSS, **lsmt.validation_metric_metadata)

    learning_rate_warmup_epochs: float = schema_utils.NonNegativeFloat(
        default=1.0, **lsmt.learning_rate_warmup_epochs_metadata
    )

    learning_rate_scaling: str = schema_utils.StringOptions(
        ["constant", "sqrt", "linear"], default="linear", **lsmt.learning_rate_scaling_metadata
    )


def get_trainer_jsonschema():
    return schema_utils.unload_jsonschema_from_marshmallow_class(TrainerConfig)
