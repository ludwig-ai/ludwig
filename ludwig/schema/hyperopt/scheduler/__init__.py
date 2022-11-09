from abc import ABC

# from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

# from dataclasses import field
# from typing import Dict


# ------------------
# To prevent direct dependency on ray import, the following static key stores are duplicated:

# from ray.tune.schedulers import SCHEDULER_IMPORT
# https://github.com/ray-project/ray/blob/137a1b12c3b31a3622fa5f721a05a64e9b559b05/python/ray/tune/schedulers/__init__.py#L28

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

# from ray.tune.result import DEFAULT_METRIC
RAY_TUNE_DESULT_DEFAULT_METRIC = "_metric"
# ------------------

scheduler_registry = Registry()


def register_optimizer(name: str):
    def wrap(scheduler_config: BaseSchedulerConfig):
        scheduler_registry[name] = scheduler_config
        return scheduler_config

    return wrap


@dataclass
class BaseSchedulerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    type: str


@dataclass
class AsyncHyperbandSchedulerConfig(BaseSchedulerConfig):
    """Asynchronous hyperband (ASHA) scheduler settings."""

    type: str = schema_utils.StringOptions(options=["async_hyperband"], default="async_hyperband", allow_none=False)

    time_attr: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS),
        default=TIME_TOTAL_S,
        allow_none=False,
        description=(
            "A training result attr to use for comparing time. Note that you can pass in something non-temporal such as"
            " training_iteration as a measure of progress, the only requirement is that the attribute should increase "
            "monotonically."
        ),
    )

    metric: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS) + [RAY_TUNE_DESULT_DEFAULT_METRIC],
        default=None,
        description=(
            "The training result objective value attribute. Stopping procedures will use this attribute. If None but a "
            "mode was passed, the ray.tune.result.DEFAULT_METRIC will be used per default."
        ),
    )

    mode: str = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        description=(
            "One of {min, max}. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    max_t: int = schema_utils.PositiveInteger(
        default=100,
        description=(
            "max time units per trial. Trials will be stopped after max_t time units (determined by time_attr) have "
            "passed."
        ),
    )

    grace_period: int = schema_utils.PositiveInteger(
        default=1,
        description=(
            "Only stop trials at least this old in time. The units are the same as the attribute named by time_attr."
        ),
    )

    reduction_factor: int = schema_utils.FloatRange(default=5, min=0, min_inclusive=False)


@dataclass
class HyperbandSchedulerConfig(BaseSchedulerConfig):
    """Asynchronous hyperband (ASHA) scheduler settings."""

    type: str = schema_utils.StringOptions(options=["async_hyperband"], default="async_hyperband", allow_none=False)

    time_attr: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS),
        default=TIME_TOTAL_S,
        allow_none=False,
        description=(
            "A training result attr to use for comparing time. Note that you can pass in something non-temporal such as"
            " training_iteration as a measure of progress, the only requirement is that the attribute should increase "
            "monotonically."
        ),
    )

    max_t: int = schema_utils.PositiveInteger(default=3600, description="")

    grace_period: int = schema_utils.PositiveInteger(default=72, description="")

    reduction_factor: int = schema_utils.FloatRange(default=5, min=0, min_inclusive=False)


# def SchedulerDataclassField(description: str, default: Dict = {}):
#     allow_none = False

#     class SchedulerMarshmallowField(fields.Field):
#         def _deserialize(self, value, attr, data, **kwargs):
#             if isinstance(value, dict):
#                 try:
#                     return SchedulerConfig.Schema().load(value)
#                 except (TypeError, ValidationError):
#                     raise ValidationError(f"Invalid params for scheduler: {value}, see SchedulerConfig class.")
#             raise ValidationError("Field should be dict")

#         def _jsonschema_type_mapping(self):
#             return {
#                 **schema_utils.unload_jsonschema_from_marshmallow_class(SchedulerConfig),
#                 "title": "scheduler",
#                 "description": description,
#             }

#     if not isinstance(default, dict):
#         raise ValidationError(f"Invalid default: `{default}`")

#     load_default = SchedulerConfig.Schema().load(default)
#     dump_default = SchedulerConfig.Schema().dump(default)

#     return field(
#         metadata={
#             "marshmallow_field": SchedulerMarshmallowField(
#                 allow_none=allow_none,
#                 load_default=load_default,
#                 dump_default=dump_default,
#                 metadata={"description": description, "parameter_metadata": None},
#             )
#         },
#         default_factory=lambda: load_default,
#     )
