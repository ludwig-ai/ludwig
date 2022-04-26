from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.constants import COMBINED, LOSS, TRAINING
from ludwig.modules.optimization_modules import (
    BaseOptimizerConfig,
    GradientClippingConfig,
    GradientClippingDataclassField,
    OptimizerDataclassField,
)
from ludwig.schema import utils


@dataclass
class TrainerConfig(utils.BaseMarshmallowConfig):
    """TrainerConfig is a dataclass that configures most of the hyperparameters used for model training."""

    optimizer: BaseOptimizerConfig = OptimizerDataclassField(
        default={"type": "adam"}, description="Parameter values for selected torch optimizer."
    )

    epochs: int = utils.PositiveInteger(
        default=100, description="Number of epochs the algorithm is intended to be run over."
    )

    regularization_lambda: float = utils.FloatRange(
        default=0.0, min=0, description="Strength of the $L2$ regularization."
    )

    regularization_type: Optional[str] = utils.RegularizerOptions(default="l2", description="Type of regularization.")

    should_shuffle: bool = utils.Boolean(
        default=True, description="Whether to shuffle batches during training when true."
    )

    learning_rate: float = utils.NumericOrStringOptionsField(
        default=0.001,
        min=0.0,
        max=1.0,
        options=["auto"],
        default_numeric=0.001,
        default_option="auto",
        nullable=False,
        description=(
            "Learning rate specified in configuration, represents how much to scale the gradients by. If 'auto', "
            "`tune_learning_rate` must be called before training to estimate the optimal learning rate."
        ),
    )

    batch_size: Union[int, str] = utils.IntegerOrStringOptionsField(
        default=128,
        options=["auto"],
        default_numeric=128,
        default_option="auto",
        nullable=False,
        min_exclusive=0,
        description="Size of batch to pass to the model for training.",
    )

    eval_batch_size: Union[None, int, str] = utils.IntegerOrStringOptionsField(
        default=None,
        options=["auto"],
        default_numeric=None,
        default_option="auto",
        nullable=True,
        min_exclusive=0,
        description="Size of batch to pass to the model for evaluation.",
    )

    early_stop: int = utils.IntegerRange(
        default=5,
        min=-1,
        description=(
            "How many epochs without any improvement in the `validation_metric` triggers the algorithm to stop. Can be "
            "set to -1, which disables `early_stop`."
        ),
    )

    steps_per_checkpoint: int = utils.NonNegativeInteger(
        default=0,
        description=(
            "How often the model is checkpointed. Also dictates maximum evaluation frequency. If 0 the model is "
            "checkpointed after every epoch."
        ),
    )

    checkpoints_per_epoch: int = utils.NonNegativeInteger(
        default=0,
        description=(
            "Number of checkpoints per epoch. For example, 2 -> checkpoints are written every half of an epoch. Note "
            "that it is invalid to specify both non-zero `steps_per_checkpoint` and non-zero `checkpoints_per_epoch`."
        ),
    )

    evaluate_training_set: bool = utils.Boolean(
        default=True, description="Whether to include the entire training set during evaluation."
    )

    reduce_learning_rate_on_plateau: float = utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description=(
            "Reduces the learning rate when the algorithm hits a plateau (i.e. the performance on the validation does "
            "not improve"
        ),
    )

    reduce_learning_rate_on_plateau_patience: int = utils.NonNegativeInteger(
        default=5, description="How many epochs have to pass before the learning rate reduces."
    )

    reduce_learning_rate_on_plateau_rate: float = utils.FloatRange(
        default=0.5, min=0.0, max=1.0, description="Rate at which we reduce the learning rate."
    )

    reduce_learning_rate_eval_metric: str = utils.String(default=LOSS, description="TODO: Document parameters.")

    reduce_learning_rate_eval_split: str = utils.String(default=TRAINING, description="TODO: Document parameters.")

    increase_batch_size_on_plateau: int = utils.NonNegativeInteger(
        default=0, description="Number to increase the batch size by on a plateau."
    )

    increase_batch_size_on_plateau_patience: int = utils.NonNegativeInteger(
        default=5, description="How many epochs to wait for before increasing the batch size."
    )

    increase_batch_size_on_plateau_rate: float = utils.NonNegativeFloat(
        default=2.0, description="Rate at which the batch size increases."
    )

    increase_batch_size_on_plateau_max: int = utils.PositiveInteger(
        default=512, description="Maximum size of the batch."
    )

    increase_batch_size_eval_metric: str = utils.String(default=LOSS, description="TODO: Document parameters.")

    increase_batch_size_eval_split: str = utils.String(default=TRAINING, description="TODO: Document parameters.")

    decay: bool = utils.Boolean(default=False, description="Turn on exponential decay of the learning rate.")

    decay_steps: int = utils.PositiveInteger(default=10000, description="TODO: Document parameters.")

    decay_rate: float = utils.FloatRange(default=0.96, min=0.0, max=1.0, description="TODO: Document parameters.")

    staircase: bool = utils.Boolean(default=False, description="Decays the learning rate at discrete intervals.")

    gradient_clipping: Optional[GradientClippingConfig] = GradientClippingDataclassField(
        description="Parameter values for gradient clipping."
    )

    # TODO(#1673): Need some more logic here for validating against output features
    validation_field: str = utils.String(
        default=COMBINED,
        description="First output feature, by default it is set as the same field of the first output feature.",
    )

    validation_metric: str = utils.String(
        default=LOSS, description="Metric used on `validation_field`, set by default to accuracy."
    )

    learning_rate_warmup_epochs: float = utils.NonNegativeFloat(
        default=1.0, description="Number of epochs to warmup the learning rate for."
    )

    learning_rate_scaling: str = utils.StringOptions(
        ["constant", "sqrt", "linear"],
        default="linear",
        description=(
            "Scale by which to increase the learning rate as the number of distributed workers increases. "
            "Traditionally the learning rate is scaled linearly with the number of workers to reflect the proportion by"
            " which the effective batch size is increased. For very large batch sizes, a softer square-root scale can "
            "sometimes lead to better model performance. If the learning rate is hand-tuned for a given number of "
            "workers, setting this value to constant can be used to disable scale-up."
        ),
    )
