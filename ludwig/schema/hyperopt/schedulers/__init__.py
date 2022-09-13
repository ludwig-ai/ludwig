from dataclasses import field
from typing import Dict

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils

# Double-check if possible to use these:
# from ray.tune.schedulers import SCHEDULER_IMPORT
# https://github.com/ray-project/ray/blob/137a1b12c3b31a3622fa5f721a05a64e9b559b05/python/ray/tune/schedulers/__init__.py#L28
#
# from ray.tune.result import DEFAULT_RESULT_KEYS

# Taken from https://github.com/ray-project/ray/blob/137a1b12c3b31a3622fa5f721a05a64e9b559b05/python/ray/tune/result.py
DEFAULT_RESULT_KEYS = (
    "training_iteration",
    "time_total_s",
    "timesteps_total",
    "mean_accuracy",
    "mean_loss",
)
TIME_TOTAL_S = "time_total_s"


@dataclass
class SchedulerConfig(schema_utils.BaseMarshmallowConfig):
    """Basic scheduler settings."""

    type: str = schema_utils.StringOptions(options=["async_hyperband"], default="async_hyperband", allow_none=False)

    time_attr: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS), default=TIME_TOTAL_S, allow_none=False
    )

    max_t: int = schema_utils.PositiveInteger(default=3600, description="")

    grace_period: int = schema_utils.PositiveInteger(default=72, description="")

    reduction_factor: int = schema_utils.FloatRange(default=5, min=0, min_inclusive=False)


def SchedulerDataclassField(description: str, default: Dict = {}):
    allow_none = False

    class SchedulerMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, dict):
                try:
                    return SchedulerConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    raise ValidationError(f"Invalid params for scheduler: {value}, see SchedulerConfig class.")
            raise ValidationError("Field should be dict")

        def _jsonschema_type_mapping(self):
            return {
                **schema_utils.unload_jsonschema_from_marshmallow_class(SchedulerConfig),
                "title": "scheduler",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = SchedulerConfig.Schema().load(default)
    dump_default = SchedulerConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": SchedulerMarshmallowField(
                allow_none=allow_none,
                load_default=load_default,
                dump_default=dump_default,
                metadata={"description": description, "parameter_metadata": None},
            )
        },
        default_factory=lambda: load_default,
    )
