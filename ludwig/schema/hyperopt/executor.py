from dataclasses import field
from typing import Dict, Optional

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.hyperopt.scheduler import BaseSchedulerConfig, SchedulerDataclassField


@dataclass
class ExecutorConfig(schema_utils.BaseMarshmallowConfig):
    """Basic executor settings."""

    type: str = schema_utils.StringOptions(options=["ray"], default="ray", allow_none=False)

    num_samples: int = schema_utils.PositiveInteger(default=10, description="")

    time_budget_s: int = schema_utils.PositiveInteger(default=3600, description="")

    cpu_resources_per_trial: int = schema_utils.PositiveInteger(default=1, description="")

    gpu_resources_per_trial: int = schema_utils.NonNegativeInteger(default=0, description="")

    kubernetes_namespace: Optional[str] = schema_utils.String(default=None, description="")

    scheduler: BaseSchedulerConfig = SchedulerDataclassField(description="")


def ExecutorDataclassField(description: str, default: Dict = {}):
    allow_none = False

    class ExecutorMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, dict):
                try:
                    return ExecutorConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    raise ValidationError(f"Invalid params for scheduler: {value}, see ExecutorConfig class.")
            raise ValidationError("Field should be dict")

        def _jsonschema_type_mapping(self):
            return {
                **schema_utils.unload_jsonschema_from_marshmallow_class(ExecutorConfig),
                "title": "executor",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = ExecutorConfig.Schema().load(default)
    dump_default = ExecutorConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": ExecutorMarshmallowField(
                allow_none=allow_none,
                load_default=load_default,
                dump_default=dump_default,
                metadata={"description": description, "parameter_metadata": None},
            )
        },
        default_factory=lambda: load_default,
    )
