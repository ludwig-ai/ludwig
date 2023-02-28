from abc import ABC
from typing import Dict

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import LOSS, TEST, TRAIN, VALIDATION
from ludwig.modules import metric_modules  # noqa: Needed to ensure that the metric registry is populated.
from ludwig.modules.metric_registry import get_metric_registry
from ludwig.schema import utils as schema_utils
from ludwig.schema.hyperopt.executor import ExecutorConfig, ExecutorDataclassField
from ludwig.schema.hyperopt.search_algorithm import BaseSearchAlgorithmConfig, SearchAlgorithmDataclassField
from ludwig.schema.metadata import HYPEROPT_METADATA


@DeveloperAPI
@dataclass
class HyperoptConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Basic hyperopt settings."""

    output_feature: str = schema_utils.String(
        default="combined",  # TODO: make more restrictive
        description=HYPEROPT_METADATA["output_feature"].short_description,
    )

    goal: str = schema_utils.StringOptions(
        options=["minimize", "maximize"],
        default="minimize",
        allow_none=False,
        description=HYPEROPT_METADATA["goal"].short_description,
    )

    metric: str = schema_utils.StringOptions(
        options=get_metric_registry().keys(),
        default=LOSS,
        allow_none=False,
        description=HYPEROPT_METADATA["metric"].short_description,
    )

    split: str = schema_utils.StringOptions(
        options=[TRAIN, VALIDATION, TEST],
        default=VALIDATION,
        allow_none=False,
        description=HYPEROPT_METADATA["split"].short_description,
    )

    search_alg: BaseSearchAlgorithmConfig = SearchAlgorithmDataclassField(
        description=HYPEROPT_METADATA["search_alg"].short_description
    )

    executor: ExecutorConfig = ExecutorDataclassField(description=HYPEROPT_METADATA["executor"].short_description)

    parameters: Dict = schema_utils.Dict(
        allow_none=False, description=HYPEROPT_METADATA["parameters"].short_description
    )


@DeveloperAPI
def get_hyperopt_jsonschema():
    props = schema_utils.unload_jsonschema_from_marshmallow_class(HyperoptConfig)["properties"]

    return {
        "type": ["object", "null"],
        "properties": props,
        "title": "hyperopt_options",
        "description": "Settings for hyperopt",
    }


@DeveloperAPI
class HyperoptField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(HyperoptConfig, default_missing=True)

    @staticmethod
    def _jsonschema_type_mapping():
        return get_hyperopt_jsonschema()
