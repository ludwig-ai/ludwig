from dataclasses import field
from typing import Dict, Optional, Union

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import RAY
from ludwig.schema import utils as schema_utils
from ludwig.schema.hyperopt.scheduler import BaseSchedulerConfig, SchedulerDataclassField
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class ExecutorConfig(schema_utils.BaseMarshmallowConfig):
    """Basic executor settings."""

    type: str = schema_utils.ProtectedString(RAY)

    num_samples: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description=(
            "This parameter, along with the space specifications in the parameters section, controls how many "
            "trials are generated."
        ),
    )

    time_budget_s: int = schema_utils.PositiveInteger(
        default=3600, allow_none=True, description="The number of seconds for the entire hyperopt run."
    )

    trial_driver_resources: Dict[str, float] = schema_utils.Dict(
        default=None,
        description=(
            "The resources reserved by each trial driver. This differs from cpu_resources_per_trial and "
            "gpu_resources_per_trial because these resources are reserved for the driver, not its subsequent "
            "workers. Only used when the trials themselves are on the Ray backend. Defaults to 1 CPU."
        ),
    )

    cpu_resources_per_trial: int = schema_utils.PositiveInteger(
        default=1, description="The number of CPU cores allocated to each trial"
    )

    gpu_resources_per_trial: int = schema_utils.NonNegativeInteger(
        default=0, description="The number of GPU devices allocated to each trial"
    )

    kubernetes_namespace: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description=(
            "When running on Kubernetes, provide the namespace of the Ray cluster to sync results between "
            "pods. See the Ray docs for more info."
        ),
    )

    max_concurrent_trials: Union[str, int, None] = schema_utils.OneOfOptionsField(
        default="auto",
        allow_none=True,
        description=("The maximum number of trials to train concurrently. Defaults to auto if not specified."),
        field_options=[
            schema_utils.PositiveInteger(
                default=1, allow_none=False, description="Manually set a number of concurrent trials."
            ),
            schema_utils.StringOptions(
                options=["auto"],
                default="auto",
                allow_none=False,
                description="Automatically set number of concurrent trials.",
            ),
        ],
    )

    scheduler: BaseSchedulerConfig = SchedulerDataclassField(description="")


@DeveloperAPI
def ExecutorDataclassField(description: str, default: Dict = {}):
    class ExecutorMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, dict):
                try:
                    return ExecutorConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    raise ValidationError(f"Invalid params for executor: {value}, see ExecutorConfig class.")
            raise ValidationError("Field should be dict")

        def _jsonschema_type_mapping(self):
            return {
                **schema_utils.unload_jsonschema_from_marshmallow_class(ExecutorConfig),
                "title": "executor",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = lambda: ExecutorConfig.Schema().load(default)
    dump_default = ExecutorConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": ExecutorMarshmallowField(
                allow_none=False,
                load_default=load_default,
                dump_default=dump_default,
                metadata={"description": description, "parameter_metadata": None},
            )
        },
        default_factory=load_default,
    )
