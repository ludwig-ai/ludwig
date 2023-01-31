from abc import ABC
from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import COMBINED, DEFAULT_BATCH_SIZE, LOSS, MAX_POSSIBLE_BATCH_SIZE, MODEL_ECD, MODEL_GBM, TRAINING
from ludwig.schema import utils as schema_utils
from ludwig.schema.lr_scheduler import LRSchedulerConfig, LRSchedulerDataclassField
from ludwig.schema.metadata import TRAINER_METADATA
from ludwig.schema.optimizers import (
    BaseOptimizerConfig,
    GradientClippingConfig,
    GradientClippingDataclassField,
    OptimizerDataclassField,
)
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

trainer_schema_registry = Registry()


@DeveloperAPI
def register_trainer_schema(model_type: str):
    def wrap(trainer_config: BaseTrainerConfig):
        trainer_schema_registry[model_type] = trainer_config
        return trainer_config

    return wrap


@DeveloperAPI
@ludwig_dataclass
class BaseTrainerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Common trainer parameter values."""

    early_stop: int = schema_utils.IntegerRange(
        default=5,
        min=-1,
        description=(
            "Number of consecutive rounds of evaluation without any improvement on the `validation_metric` that "
            "triggers training to stop. Can be set to -1, which disables early stopping entirely."
        ),
        parameter_metadata=TRAINER_METADATA["early_stop"],
    )

    early_stop_timeout_s: float = schema_utils.FloatRange(
        default=-1,
        min=-1,
        description=(
            "Maximum number of seconds to wait for an improvement in the `validation_metric` before stopping "
            "training. Can be set to -1, which disables early stopping based on time."
        ),
        parameter_metadata=TRAINER_METADATA["early_stop_timeout_s"],
    )

    early_stop_timeout_steps: int = schema_utils.IntegerRange(
        default=-1,
        min=-1,
        description=(
            "In case of early stopping based on time, this parameter controls how many steps of no improvement "
            "to allow before stopping training. Can be set to -1, which ignores this parameter. "
            "The training will timeout after `early_stop_timeout_s` seconds (if set) but will allow for "
            "`early_stop_timeout_steps` steps (if set) of no improvement in the `validation_metric` before "
            "stopping training."
        ),
    )


@DeveloperAPI
@register_trainer_schema(MODEL_ECD)
@dataclass(order=True)
class ECDTrainerConfig(BaseTrainerConfig):
    """Dataclass that configures most of the hyperparameters used for ECD model training."""

    learning_rate: Union[float, str] = schema_utils.OneOfOptionsField(
        default=0.001,
        allow_none=False,
        description=(
            "Controls how much to change the model in response to the estimated error each time the model weights are "
            "updated. If 'auto', the optimal learning rate is estimated by choosing the learning rate that produces "
            "the smallest non-diverging gradient update."
        ),
        parameter_metadata=TRAINER_METADATA["learning_rate"],
        field_options=[
            schema_utils.FloatRange(default=0.001, allow_none=False, min=0, max=1),
            schema_utils.StringOptions(options=["auto"], default="auto", allow_none=False),
        ],
    )

    learning_rate_scheduler: LRSchedulerConfig = LRSchedulerDataclassField(
        description="Parameter values for learning rate scheduler.",
        default=None,
    )

    epochs: int = schema_utils.PositiveInteger(
        default=100,
        description="Number of epochs the algorithm is intended to be run over. Overridden if `train_steps` is set",
        parameter_metadata=TRAINER_METADATA["epochs"],
    )

    checkpoints_per_epoch: int = schema_utils.NonNegativeInteger(
        default=0,
        description=(
            "Number of checkpoints per epoch. For example, 2 -> checkpoints are written every half of an epoch. Note "
            "that it is invalid to specify both non-zero `steps_per_checkpoint` and non-zero `checkpoints_per_epoch`."
        ),
        parameter_metadata=TRAINER_METADATA["checkpoints_per_epoch"],
    )

    train_steps: int = schema_utils.PositiveInteger(
        default=None,
        description=(
            "Maximum number of training steps the algorithm is intended to be run over. Unset by default. "
            "If set, will override `epochs` and if left unset then `epochs` is used to determine training length."
        ),
        parameter_metadata=TRAINER_METADATA["train_steps"],
    )

    steps_per_checkpoint: int = schema_utils.NonNegativeInteger(
        default=0,
        description=(
            "How often the model is checkpointed. Also dictates maximum evaluation frequency. If 0 the model is "
            "checkpointed after every epoch."
        ),
        parameter_metadata=TRAINER_METADATA["steps_per_checkpoint"],
    )

    batch_size: Union[int, str] = schema_utils.OneOfOptionsField(
        default=DEFAULT_BATCH_SIZE,
        allow_none=False,
        description=(
            "The number of training examples utilized in one training step of the model. If ’auto’, the "
            "batch size that maximized training throughput (samples / sec) will be used. For CPU training, the "
            "tuned batch size is capped at 128 as throughput benefits of large batch sizes are less noticeable without "
            "a GPU."
        ),
        parameter_metadata=TRAINER_METADATA["batch_size"],
        field_options=[
            schema_utils.PositiveInteger(default=128, description="", allow_none=False),
            schema_utils.StringOptions(options=["auto"], default="auto", allow_none=False),
        ],
    )

    max_batch_size: int = schema_utils.PositiveInteger(
        default=MAX_POSSIBLE_BATCH_SIZE,
        allow_none=True,
        description=(
            "Auto batch size tuning and increasing batch size on plateau will be capped at this value. The default "
            "value is 2^40."
        ),
        parameter_metadata=TRAINER_METADATA["max_batch_size"],
    )

    early_stop: int = schema_utils.IntegerRange(
        default=5,
        min=-1,
        description=(
            "Number of consecutive rounds of evaluation without any improvement on the `validation_metric` that "
            "triggers training to stop. Can be set to -1, which disables early stopping entirely."
        ),
        parameter_metadata=TRAINER_METADATA["early_stop"],
    )

    eval_batch_size: Union[None, int, str] = schema_utils.OneOfOptionsField(
        default=None,
        description=(
            "Size of batch to pass to the model for evaluation. If it is `0` or `None`, the same value of `batch_size` "
            "is used. This is useful to speedup evaluation with a much bigger batch size than training, if enough "
            "memory is available. If ’auto’, the biggest batch size (power of 2) that can fit in memory will be used."
        ),
        parameter_metadata=TRAINER_METADATA["eval_batch_size"],
        field_options=[
            schema_utils.PositiveInteger(default=128, description="", allow_none=False),
            schema_utils.StringOptions(options=["auto"], default="auto", allow_none=False),
        ],
    )

    evaluate_training_set: bool = schema_utils.Boolean(
        default=False,
        description=(
            "Whether to evaluate on the entire training set during evaluation. By default, training metrics will be "
            "computed at the end of each training step, and accumulated up to the evaluation phase. In practice, "
            "computing training set metrics during training is up to 30% faster than running a separate evaluation "
            "pass over the training set, but results in more noisy training metrics, particularly during the earlier "
            "epochs. It's recommended to only set this to True if you need very exact training set metrics, and are "
            "willing to pay a significant performance penalty for them."
        ),
        parameter_metadata=TRAINER_METADATA["evaluate_training_set"],
    )

    validation_field: str = schema_utils.String(
        default=None,
        description="The field for which the `validation_metric` is used for validation-related mechanics like early "
        "stopping, parameter change plateaus, as well as what hyperparameter optimization uses to determine the best "
        "trial. If unset (default), the first output feature is used. If explicitly specified, neither "
        "`validation_field` nor `validation_metric` are overwritten.",
        parameter_metadata=TRAINER_METADATA["validation_field"],
    )

    validation_metric: str = schema_utils.String(
        default=LOSS,
        description=(
            "Metric from `validation_field` that is used. If validation_field is not explicitly specified, this is "
            "overwritten to be the first output feature type's `default_validation_metric`, consistent with "
            "validation_field. If the validation_metric is specified, then we will use the first output feature that "
            "produces this metric as the `validation_field`."
        ),
        parameter_metadata=TRAINER_METADATA["validation_metric"],
    )

    optimizer: BaseOptimizerConfig = OptimizerDataclassField(
        default={"type": "adam"}, description="Parameter values for selected torch optimizer."
    )

    regularization_type: Optional[str] = schema_utils.RegularizerOptions(
        default="l2",
        description="Type of regularization.",
        parameter_metadata=TRAINER_METADATA["regularization_type"],
    )

    regularization_lambda: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Strength of the $L2$ regularization.",
        parameter_metadata=TRAINER_METADATA["regularization_lambda"],
    )

    should_shuffle: bool = schema_utils.Boolean(
        default=True,
        description="Whether to shuffle batches during training when true.",
        parameter_metadata=TRAINER_METADATA["should_shuffle"],
    )

    increase_batch_size_on_plateau: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of times to increase the batch size on a plateau.",
        parameter_metadata=TRAINER_METADATA["increase_batch_size_on_plateau"],
    )

    increase_batch_size_on_plateau_patience: int = schema_utils.NonNegativeInteger(
        default=5,
        description="How many epochs to wait for before increasing the batch size.",
        parameter_metadata=TRAINER_METADATA["increase_batch_size_on_plateau_patience"],
    )

    increase_batch_size_on_plateau_rate: float = schema_utils.NonNegativeFloat(
        default=2.0,
        description="Rate at which the batch size increases.",
        parameter_metadata=TRAINER_METADATA["increase_batch_size_on_plateau_rate"],
    )

    increase_batch_size_eval_metric: str = schema_utils.String(
        default=LOSS,
        description="Which metric to listen on for increasing the batch size.",
        parameter_metadata=TRAINER_METADATA["increase_batch_size_eval_metric"],
    )

    increase_batch_size_eval_split: str = schema_utils.String(
        default=TRAINING,
        description="Which dataset split to listen on for increasing the batch size.",
        parameter_metadata=TRAINER_METADATA["increase_batch_size_eval_split"],
    )

    gradient_clipping: Optional[GradientClippingConfig] = GradientClippingDataclassField(
        description="Parameter values for gradient clipping.",
        default={},
    )

    learning_rate_scaling: str = schema_utils.StringOptions(
        ["constant", "sqrt", "linear"],
        default="linear",
        description="Scale by which to increase the learning rate as the number of distributed workers increases. "
        "Traditionally the learning rate is scaled linearly with the number of workers to reflect the "
        "proportion by"
        " which the effective batch size is increased. For very large batch sizes, a softer square-root "
        "scale can "
        "sometimes lead to better model performance. If the learning rate is hand-tuned for a given "
        "number of "
        "workers, setting this value to constant can be used to disable scale-up.",
        parameter_metadata=TRAINER_METADATA["learning_rate_scaling"],
    )

    bucketing_field: str = schema_utils.String(
        default=None,
        description="Feature to use for bucketing datapoints",
        parameter_metadata=TRAINER_METADATA["bucketing_field"],
    )

    use_mixed_precision: bool = schema_utils.Boolean(
        default=False,
        description="Enable automatic mixed-precision (AMP) during training.",
        parameter_metadata=TRAINER_METADATA["use_mixed_precision"],
    )


@DeveloperAPI
@register_trainer_schema(MODEL_GBM)
@ludwig_dataclass
class GBMTrainerConfig(BaseTrainerConfig):
    """Dataclass that configures most of the hyperparameters used for GBM model training."""

    # NOTE: Overwritten here since GBM performs better with a different default learning rate.
    learning_rate: Union[float, str] = schema_utils.NonNegativeFloat(
        default=0.03,
        allow_none=False,
        description=(
            "Controls how much to change the model in response to the estimated error each time the model weights are "
            "updated."
        ),
        parameter_metadata=TRAINER_METADATA["learning_rate"],
    )

    # LightGBM Learning Control params
    max_depth: int = schema_utils.Integer(
        default=18,
        description="Maximum depth of a tree in the GBM trainer. A negative value means no limit.",
    )

    drop_rate: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="Dropout rate for the GBM trainer. Used only with boosting_type 'dart'.",
    )

    # NOTE: Overwritten here to provide a default value. In many places, we fall back to eval_batch_size if batch_size
    # is not specified. GBM does not have a value for batch_size, so we need to specify eval_batch_size here.
    eval_batch_size: Union[None, int, str] = schema_utils.PositiveInteger(
        default=1048576,
        description="Size of batch to pass to the model for evaluation.",
        parameter_metadata=TRAINER_METADATA["eval_batch_size"],
    )

    evaluate_training_set: bool = schema_utils.Boolean(
        default=False,
        description=(
            "Whether to evaluate on the entire training set during evaluation. By default, training metrics will be "
            "computed at the end of each training step, and accumulated up to the evaluation phase. In practice, "
            "computing training set metrics during training is up to 30% faster than running a separate evaluation "
            "pass over the training set, but results in more noisy training metrics, particularly during the earlier "
            "epochs. It's recommended to only set this to True if you need very exact training set metrics, and are "
            "willing to pay a significant performance penalty for them."
        ),
        parameter_metadata=TRAINER_METADATA["evaluate_training_set"],
    )

    # TODO(#1673): Need some more logic here for validating against output features
    validation_field: str = schema_utils.String(
        default=COMBINED,
        description="First output feature, by default it is set as the same field of the first output feature.",
        parameter_metadata=TRAINER_METADATA["validation_field"],
    )

    validation_metric: str = schema_utils.String(
        default=LOSS,
        description=(
            "Metric used on `validation_field`, set by default to the "
            "output feature type's `default_validation_metric`."
        ),
        parameter_metadata=TRAINER_METADATA["validation_metric"],
    )

    tree_learner: str = schema_utils.StringOptions(
        ["serial", "feature", "data", "voting"],
        allow_none=False,
        default="serial",
        description="Type of tree learner to use with GBM trainer.",
    )

    # LightGBM core parameters (https://lightgbm.readthedocs.io/en/latest/Parameters.html)
    boosting_type: str = schema_utils.StringOptions(
        # TODO: Re-enable "goss" when supported: https://github.com/ludwig-ai/ludwig/issues/2988
        ["gbdt", "dart"],
        allow_none=False,
        default="gbdt",
        description="Type of boosting algorithm to use with GBM trainer.",
    )

    boosting_rounds_per_checkpoint: int = schema_utils.PositiveInteger(
        default=50, description="Number of boosting rounds per checkpoint / evaluation round."
    )

    num_boost_round: int = schema_utils.PositiveInteger(
        default=1000, description="Number of boosting rounds to perform with GBM trainer."
    )

    num_leaves: int = schema_utils.PositiveInteger(
        default=82, description="Number of leaves to use in the tree with GBM trainer."
    )

    min_data_in_leaf: int = schema_utils.NonNegativeInteger(
        default=20, description="Minimum number of data points in a leaf with GBM trainer."
    )

    min_sum_hessian_in_leaf: float = schema_utils.NonNegativeFloat(
        default=1e-3, description="Minimum sum of hessians in a leaf with GBM trainer."
    )

    bagging_fraction: float = schema_utils.FloatRange(
        default=0.8, min=0, max=1, description="Fraction of data to use for bagging with GBM trainer."
    )

    pos_bagging_fraction: float = schema_utils.FloatRange(
        default=1.0, min=0, max=1, description="Fraction of positive data to use for bagging with GBM trainer."
    )

    neg_bagging_fraction: float = schema_utils.FloatRange(
        default=1.0, min=0, max=1, description="Fraction of negative data to use for bagging with GBM trainer."
    )

    bagging_freq: int = schema_utils.NonNegativeInteger(default=1, description="Frequency of bagging with GBM trainer.")

    bagging_seed: int = schema_utils.Integer(default=3, description="Random seed for bagging with GBM trainer.")

    feature_fraction: float = schema_utils.FloatRange(
        default=0.75, min=0, max=1, description="Fraction of features to use in the GBM trainer."
    )

    feature_fraction_bynode: float = schema_utils.FloatRange(
        default=1.0, min=0, max=1, description="Fraction of features to use for each tree node with GBM trainer."
    )

    feature_fraction_seed: int = schema_utils.Integer(
        default=2, description="Random seed for feature fraction with GBM trainer."
    )

    extra_trees: bool = schema_utils.Boolean(
        default=False, description="Whether to use extremely randomized trees in the GBM trainer."
    )

    extra_seed: int = schema_utils.Integer(
        default=6, description="Random seed for extremely randomized trees in the GBM trainer."
    )

    max_delta_step: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description=(
            "Used to limit the max output of tree leaves in the GBM trainer. A negative value means no constraint."
        ),
    )

    lambda_l1: float = schema_utils.NonNegativeFloat(
        default=0.25, description="L1 regularization factor for the GBM trainer."
    )

    lambda_l2: float = schema_utils.NonNegativeFloat(
        default=0.2, description="L2 regularization factor for the GBM trainer."
    )

    linear_lambda: float = schema_utils.NonNegativeFloat(
        default=0.0, description="Linear tree regularization in the GBM trainer."
    )

    min_gain_to_split: float = schema_utils.NonNegativeFloat(
        default=0.03, description="Minimum gain to split a leaf in the GBM trainer."
    )

    max_drop: int = schema_utils.Integer(
        default=50,
        description=(
            "Maximum number of dropped trees during one boosting iteration. "
            "Used only with boosting_type 'dart'. A negative value means no limit."
        ),
    )

    skip_drop: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description=(
            "Probability of skipping the dropout during one boosting iteration. Used only with boosting_type 'dart'."
        ),
    )

    xgboost_dart_mode: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use xgboost dart mode in the GBM trainer. Used only with boosting_type 'dart'.",
    )

    uniform_drop: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use uniform dropout in the GBM trainer. Used only with boosting_type 'dart'.",
    )

    drop_seed: int = schema_utils.Integer(
        default=4,
        description="Random seed to choose dropping models in the GBM trainer. Used only with boosting_type 'dart'.",
    )

    top_rate: float = schema_utils.FloatRange(
        default=0.2,
        min=0,
        max=1,
        description="The retain ratio of large gradient data in the GBM trainer. Used only with boosting_type 'goss'.",
    )

    other_rate: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The retain ratio of small gradient data in the GBM trainer. Used only with boosting_type 'goss'.",
    )

    min_data_per_group: int = schema_utils.PositiveInteger(
        default=100,
        description="Minimum number of data points per categorical group for the GBM trainer.",
    )

    max_cat_threshold: int = schema_utils.PositiveInteger(
        default=32,
        description="Number of split points considered for categorical features for the GBM trainer.",
    )

    cat_l2: float = schema_utils.NonNegativeFloat(
        default=10.0, description="L2 regularization factor for categorical split in the GBM trainer."
    )

    cat_smooth: float = schema_utils.NonNegativeFloat(
        default=10.0, description="Smoothing factor for categorical split in the GBM trainer."
    )

    max_cat_to_onehot: int = schema_utils.PositiveInteger(
        default=4,
        description="Maximum categorical cardinality required before one-hot encoding in the GBM trainer.",
    )

    cegb_tradeoff: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Cost-effective gradient boosting multiplier for all penalties in the GBM trainer.",
    )

    cegb_penalty_split: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Cost-effective gradient boosting penalty for splitting a node in the GBM trainer.",
    )

    path_smooth: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Smoothing factor applied to tree nodes in the GBM trainer.",
    )

    verbose: int = schema_utils.IntegerOptions(
        options=list(range(-1, 3)), allow_none=False, default=-1, description="Verbosity level for GBM trainer."
    )

    # LightGBM IO params
    max_bin: int = schema_utils.PositiveInteger(
        default=255, description="Maximum number of bins to use for discretizing features with GBM trainer."
    )

    feature_pre_filter: bool = schema_utils.Boolean(
        default=True,
        description="Whether to ignore features that are unsplittable based on min_data_in_leaf in the GBM trainer.",
    )


@DeveloperAPI
def get_model_type_jsonschema(model_type: str = MODEL_ECD):
    enum = [MODEL_ECD]
    if model_type == MODEL_GBM:
        enum = [MODEL_GBM]

    return {
        "type": "string",
        "enum": enum,
        "default": MODEL_ECD,
        "title": "model_type",
        "description": "Select the model type.",
    }


@DeveloperAPI
def get_trainer_jsonschema(model_type: str):
    trainer_cls = trainer_schema_registry[model_type]
    props = schema_utils.unload_jsonschema_from_marshmallow_class(trainer_cls)["properties"]

    return {
        "type": "object",
        "properties": props,
        "title": "trainer_options",
        "additionalProperties": False,
        "description": "Schema for trainer determined by Model Type",
    }
