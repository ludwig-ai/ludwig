from abc import ABC
from dataclasses import field
from typing import Dict, Optional

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass
from ludwig.constants import LOSS, TRAINING
from ludwig.schema.metadata.trainer_metadata import TRAINER_METADATA

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI


@DeveloperAPI
@dataclass(repr=False)
class LRSchedulerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Configuration for learning rate scheduler parameters."""

    learning_rate_warmup_fraction: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Fraction of total training steps to warmup the learning rate for.",
        parameter_metadata=TRAINER_METADATA["learning_rate_warmup_fraction"],
    )

    learning_rate_warmup_evaluations: int = schema_utils.NonNegativeFloat(
        default=0,
        description="Number of evaluation steps to warmup the learning rate for.",
        parameter_metadata=TRAINER_METADATA["learning_rate_warmup_evaluations"],
    )

    decay: Optional[str] = schema_utils.StringOptions(
        ["linear", "exponential"],
        description="Turn on decay of the learning rate.",
        parameter_metadata=TRAINER_METADATA["decay"],
    )

    decay_steps: int = schema_utils.PositiveInteger(
        default=10000,
        description="The number of steps to take in the exponential learning rate decay.",
        parameter_metadata=TRAINER_METADATA["decay_steps"],
    )

    decay_rate: float = schema_utils.FloatRange(
        default=0.96,
        min=0,
        max=1,
        description="Decay per epoch (%): Factor to decrease the Learning rate.",
        parameter_metadata=TRAINER_METADATA["decay_steps"],
    )

    staircase: bool = schema_utils.Boolean(
        default=False,
        description="Decays the learning rate at discrete intervals.",
        parameter_metadata=TRAINER_METADATA["staircase"],
    )

    reduce_learning_rate_on_plateau: int = schema_utils.NonNegativeInteger(
        default=0,
        description=(
            "How many times to reduce the learning rate when the algorithm hits a plateau (i.e. the performance on the"
            "training set does not improve"
        ),
        parameter_metadata=TRAINER_METADATA["reduce_learning_rate_on_plateau"],
    )

    reduce_learning_rate_on_plateau_patience: int = schema_utils.NonNegativeInteger(
        default=5,
        description=(
            "How many evaluation steps have to pass before the learning rate reduces "
            "when `reduce_learning_rate_on_plateau > 0`."
        ),
        parameter_metadata=TRAINER_METADATA["reduce_learning_rate_on_plateau_patience"],
    )

    reduce_learning_rate_on_plateau_rate: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description="Rate at which we reduce the learning rate when `reduce_learning_rate_on_plateau > 0`.",
        parameter_metadata=TRAINER_METADATA["reduce_learning_rate_on_plateau_rate"],
    )

    reduce_learning_rate_eval_metric: str = schema_utils.String(
        default=LOSS,
        description="Rate at which we reduce the learning rate when `reduce_learning_rate_on_plateau > 0`.",
        parameter_metadata=TRAINER_METADATA["reduce_learning_rate_eval_metric"],
    )

    reduce_learning_rate_eval_split: str = schema_utils.String(
        default=TRAINING,
        description=(
            "Which dataset split to listen on for reducing the learning rate "
            "when `reduce_learning_rate_on_plateau > 0`."
        ),
        parameter_metadata=TRAINER_METADATA["reduce_learning_rate_eval_split"],
    )


# TODO(travis): too much boilerplate here, we should find a way to abstract all this and only require specifying the
# minimal amount needed for the new config object.
@DeveloperAPI
def LRSchedulerDataclassField(description: str, default: Dict = None):
    """Returns custom dataclass field for `LRSchedulerConfig`. Allows `None` by default.

    :param description: Description of the dataclass field
    :param default: dict that specifies param values that will be loaded by its schema class (default: None).
    """
    allow_none = True
    default = default or {}

    class LRSchedulerMarshmallowField(fields.Field):
        """Custom marshmallow field class for learjing rate scheduler.

        Deserializes a dict to a valid instance of `LRSchedulerConfig` and
        creates a corresponding JSON schema for external usage.
        """

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return value
            if isinstance(value, dict):
                try:
                    return LRSchedulerConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    # TODO(travis): this seems much too verbose, does the validation error not show the specific error?
                    raise ValidationError(
                        f"Invalid params for learning rate scheduler: {value}, see LRSchedulerConfig class."
                    )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "oneOf": [
                    {"type": "null", "title": "disabled", "description": "Disable learning rate scheduler."},
                    {
                        **schema_utils.unload_jsonschema_from_marshmallow_class(LRSchedulerConfig),
                        "title": "enabled_options",
                    },
                ],
                "title": "learning_rate_scheduler_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = LRSchedulerConfig.Schema().load(default)
    dump_default = LRSchedulerConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": LRSchedulerMarshmallowField(
                allow_none=allow_none,
                load_default=load_default,
                dump_default=dump_default,
                metadata={
                    "description": description,
                    # TODO(travis): do this once we convert the metadata to yaml so it's not so painful
                    # "parameter_metadata": convert_metadata_to_json(TRAINER_METADATA["learning_rate_scheduler"]),
                },
            )
        },
        default_factory=lambda: load_default,
    )
