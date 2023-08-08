from abc import ABC
from dataclasses import field
from typing import Dict

from marshmallow import fields, ValidationError

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import LOSS, MODEL_ECD, TRAINING
from ludwig.schema.metadata import TRAINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class LRSchedulerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Configuration for learning rate scheduler parameters."""

    decay: str = schema_utils.StringOptions(
        options=["linear", "exponential", "cosine"],
        default=None,
        allow_none=True,
        description="Turn on decay of the learning rate.",
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
            "How many evaluation steps have to pass before the learning rate reduces " "when `reduce_on_plateau > 0`."
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
        description=(
            "Metric plateau used to trigger when we reduce the learning rate " "when `reduce_on_plateau > 0`."
        ),
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["learning_rate_scheduler"]["reduce_eval_metric"],
    )

    reduce_eval_split: str = schema_utils.String(
        default=TRAINING,
        allow_none=False,
        description=(
            "Which dataset split to listen on for reducing the learning rate " "when `reduce_on_plateau > 0`."
        ),
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


# TODO(travis): too much boilerplate here, we should find a way to abstract all this and only require specifying the
# minimal amount needed for the new config object.
@DeveloperAPI
def LRSchedulerDataclassField(description: str, default: Dict = None):
    """Returns custom dataclass field for `LRSchedulerConfig`. Allows `None` by default.

    Args:
        description: Description of the dataclass field
        default: dict that specifies param values that will be loaded by its schema class (default: None).
    """
    allow_none = True
    default = default or {}

    class LRSchedulerMarshmallowField(fields.Field):
        """Custom marshmallow field class for learjing rate scheduler.

        Deserializes a dict to a valid instance of `LRSchedulerConfig` and creates a corresponding JSON schema for
        external usage.
        """

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return value
            if isinstance(value, dict):
                try:
                    return LRSchedulerConfig.Schema().load(value)
                except (TypeError, ValidationError) as e:
                    # TODO(travis): this seems much too verbose, does the validation error not show the specific error?
                    raise ValidationError(
                        f"Invalid params for learning rate scheduler: {value}, see LRSchedulerConfig class. Error: {e}"
                    )
            raise ValidationError("Field should be None or dict")

        def _jsonschema_type_mapping(self):
            return {
                **schema_utils.unload_jsonschema_from_marshmallow_class(LRSchedulerConfig),
                "title": "learning_rate_scheduler_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = lambda: LRSchedulerConfig.Schema().load(default)
    dump_default = LRSchedulerConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": LRSchedulerMarshmallowField(
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
