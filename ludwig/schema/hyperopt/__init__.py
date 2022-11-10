from abc import ABC
from typing import Any

from marshmallow_dataclass import dataclass

from ludwig.constants import LOSS
from ludwig.features.feature_registries import output_type_registry
from ludwig.schema import utils as schema_utils
from ludwig.schema.hyperopt.executor import ExecutorConfig, ExecutorDataclassField
from ludwig.schema.hyperopt.search_algorithm import SearchAlgorithmConfig, SearchAlgorithmDataclassField


def get_hyperopt_metric_options():
    all_metrics = []
    for oftype in output_type_registry:
        ofcls = output_type_registry[oftype]
        all_metrics += ofcls.metric_functions.keys()
    return all_metrics


@dataclass
class HyperoptConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Basic hyperopt settings."""

    output_feature: str = "combined"  # TODO: make more restrictive

    goal: str = schema_utils.StringOptions(options=["minimize", "maximize"], default="minimize", allow_none=False)

    metric: str = schema_utils.StringOptions(options=get_hyperopt_metric_options(), default=LOSS, allow_none=False)

    search_alg: SearchAlgorithmConfig = SearchAlgorithmDataclassField(description="")

    executor: ExecutorConfig = ExecutorDataclassField(description="")

    parameters: Any = None


def get_hyperopt_jsonschema():
    props = schema_utils.unload_jsonschema_from_marshmallow_class(HyperoptConfig)["properties"]

    return {
        "type": "object",
        "properties": props,
        "title": "hyperopt_options",
        "description": "Schema for hyperopt",
    }
