from abc import ABC
from dataclasses import field

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import LOSS, MODEL_ECD, TRAINING
from ludwig.error import ConfigValidationError
from ludwig.schema.metadata import TRAINER_METADATA


@DeveloperAPI
class LRSchedulerConfig(schema_utils.LudwigBaseConfig, ABC):
    """Configuration for learning rate scheduler parameters."""

    decay: str = schema_utils.StringOptions(
        options=["linear", "exponential", "cosine", "one_cycle", "inverse_sqrt", "polynomial", "wsd"],
        default=None,
        allow_none=True,
        description=(
            "Learning rate decay schedule. Options: 'linear', 'exponential', 'cosine', 'one_cycle', "
            "'inverse_sqrt', 'polynomial', 'wsd'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["decay"],
    )

    decay_rate: float = schema_utils.FloatRange(
        default=0.96,
        min=0,
        max=1,
        description="Decay per epoch (%): Factor to decrease the Learning rate.",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["decay_rate"],
    )

    decay_steps: int = schema_utils.PositiveInteger(
        default=10000,
        description="The number of steps to take in the exponential learning rate decay.",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["decay_steps"],
    )

    staircase: bool = schema_utils.Boolean(
        default=False,
        description="Decays the learning rate at discrete intervals.",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["staircase"],
    )

    reduce_on_plateau: int = schema_utils.NonNegativeInteger(
        default=0,
        description=(
            "How many times to reduce the learning rate when the algorithm hits a plateau (i.e. the performance on the "
            "training set does not improve)"
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["reduce_on_plateau"],
    )

    reduce_on_plateau_patience: int = schema_utils.NonNegativeInteger(
        default=10,
        description=(
            "How many evaluation steps have to pass before the learning rate reduces when `reduce_on_plateau > 0`."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["reduce_on_plateau_patience"],
    )

    reduce_on_plateau_rate: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="Rate at which we reduce the learning rate when `reduce_on_plateau > 0`.",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["reduce_on_plateau_rate"],
    )

    warmup_evaluations: int = schema_utils.NonNegativeFloat(
        default=0,
        description="Number of evaluation steps to warmup the learning rate for.",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["warmup_evaluations"],
    )

    warmup_fraction: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Fraction of total training steps to warmup the learning rate for.",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["warmup_fraction"],
    )

    reduce_eval_metric: str = schema_utils.String(
        default=LOSS,
        allow_none=False,
        description=("Metric plateau used to trigger when we reduce the learning rate when `reduce_on_plateau > 0`."),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["reduce_eval_metric"],
    )

    reduce_eval_split: str = schema_utils.String(
        default=TRAINING,
        allow_none=False,
        description=("Which dataset split to listen on for reducing the learning rate when `reduce_on_plateau > 0`."),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["reduce_eval_split"],
    )

    # Parameters for CosineAnnealingWarmRestarts scheduler

    t_0: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of steps before the first restart for cosine annealing decay. If not specified, it"
        " will be set to `steps_per_checkpoint`.",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["t_0"],
    )

    t_mult: int = schema_utils.PositiveInteger(
        default=1,
        description="Period multiplier after each restart for cosine annealing decay. Defaults to 1, i.e.,"
        " restart every `t_0` steps. If set to a larger value, the period between restarts increases by that"
        " multiplier. For e.g., if t_mult is 2, then the periods would be: t_0, 2*t_0, 2^2*t_0, 2^3*t_0, etc.",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["t_mult"],
    )

    eta_min: float = schema_utils.FloatRange(
        default=0,
        min=0,
        max=1,
        description="Minimum learning rate allowed for cosine annealing decay. Default: 0.",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["eta_min"],
    )

    # Parameters for OneCycleLR scheduler

    max_lr: float = schema_utils.Float(
        default=None,
        allow_none=True,
        description=(
            "Maximum learning rate for the OneCycleLR scheduler. If None, defaults to the optimizer's base "
            "learning rate. Used only when decay='one_cycle'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["max_lr"],
    )

    pct_start: float = schema_utils.FloatRange(
        default=0.3,
        min=0,
        max=1,
        description=(
            "Fraction of training steps spent increasing the learning rate in the OneCycleLR scheduler. "
            "Used only when decay='one_cycle'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["pct_start"],
    )

    div_factor: float = schema_utils.Float(
        default=25.0,
        allow_none=False,
        description=(
            "Determines the initial learning rate (initial_lr = max_lr / div_factor) for OneCycleLR. "
            "Used only when decay='one_cycle'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["div_factor"],
    )

    final_div_factor: float = schema_utils.Float(
        default=1e4,
        allow_none=False,
        description=(
            "Determines the minimum learning rate (min_lr = initial_lr / final_div_factor) for OneCycleLR. "
            "Used only when decay='one_cycle'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["final_div_factor"],
    )

    # Parameters for InverseSqrtLR scheduler

    inverse_sqrt_warmup_steps: int = schema_utils.PositiveInteger(
        default=4000,
        description=(
            "Number of warmup steps for the inverse square root scheduler. After warmup, the LR decays as "
            "1/sqrt(step). This is the classic Transformer schedule from Vaswani et al. (2017). "
            "Used only when decay='inverse_sqrt'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["inverse_sqrt_warmup_steps"],
    )

    # Parameters for Polynomial Decay scheduler

    polynomial_power: float = schema_utils.Float(
        default=1.0,
        allow_none=False,
        description=(
            "Power of the polynomial decay. power=1.0 gives linear decay; higher values give more concave "
            "decay curves. Used only when decay='polynomial'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["polynomial_power"],
    )

    polynomial_end_lr: float = schema_utils.Float(
        default=0.0,
        allow_none=False,
        description=(
            "Final (minimum) learning rate at the end of polynomial decay. Used only when decay='polynomial'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["polynomial_end_lr"],
    )

    # Parameters for Warmup-Stable-Decay (WSD) scheduler

    wsd_warmup_fraction: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description=(
            "Fraction of total training steps spent in the linear warmup phase of the WSD scheduler. "
            "Used only when decay='wsd'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["wsd_warmup_fraction"],
    )

    wsd_stable_fraction: float = schema_utils.FloatRange(
        default=0.8,
        min=0,
        max=1,
        description=(
            "Fraction of total training steps spent in the constant LR phase of the WSD scheduler. "
            "Used only when decay='wsd'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["wsd_stable_fraction"],
    )

    wsd_decay_fraction: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description=(
            "Fraction of total training steps spent in the decay phase of the WSD scheduler. "
            "wsd_warmup_fraction + wsd_stable_fraction + wsd_decay_fraction should sum to 1. "
            "Used only when decay='wsd'."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["wsd_decay_fraction"],
    )


# TODO(travis): too much boilerplate here, we should find a way to abstract all this and only require specifying the
# minimal amount needed for the new config object.
@DeveloperAPI
def LRSchedulerDataclassField(description: str, default: dict = None):
    """Returns custom dataclass field for `LRSchedulerConfig`. Allows `None` by default.

    Args:
        description: Description of the dataclass field
        default: dict that specifies param values that will be loaded by its schema class (default: None).
    """
    allow_none = True
    default = default or {}

    class LRSchedulerConfigField(schema_utils.SchemaField):
        """Custom field class for learning rate scheduler.

        Deserializes a dict to a valid instance of `LRSchedulerConfig` and creates a corresponding JSON schema for
        external usage.
        """

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return value
            if isinstance(value, dict):
                try:
                    return LRSchedulerConfig.model_validate(value)
                except (TypeError, ConfigValidationError) as e:
                    raise ConfigValidationError(
                        f"Invalid params for learning rate scheduler: {value}, see LRSchedulerConfig class. Error: {e}"
                    )
            raise ConfigValidationError("Field should be None or dict")

        def _jsonschema_type_mapping(self):
            return {
                **schema_utils.unload_jsonschema_from_config_class(LRSchedulerConfig),
                "title": "learning_rate_scheduler_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ConfigValidationError(f"Invalid default: `{default}`")

    load_default = lambda: LRSchedulerConfig.model_validate(default)
    try:
        dump_default = LRSchedulerConfig.model_validate(default).to_dict()
    except Exception:
        dump_default = default if isinstance(default, dict) else {}

    return field(
        metadata={
            "marshmallow_field": LRSchedulerConfigField(
                allow_none=allow_none,
                load_default=load_default,
                dump_default=dump_default,
                metadata={
                    "description": description,
                },
            )
        },
        default_factory=load_default,
    )
