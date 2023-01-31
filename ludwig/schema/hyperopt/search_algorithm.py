from dataclasses import field
from typing import Dict

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.hyperopt.registry import search_algorithm_registry  # Double-check this implicit import.
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BaseSearchAlgorithmConfig(schema_utils.BaseMarshmallowConfig):
    """Basic search algorithm settings."""

    type: str = schema_utils.StringOptions(
        options=list(search_algorithm_registry.keys()), default="hyperopt", allow_none=False
    )


@DeveloperAPI
def SearchAlgorithmDataclassField(description: str = "", default: Dict = {"type": "variant_generator"}):
    class SearchAlgorithmMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, dict):
                try:
                    return BaseSearchAlgorithmConfig.Schema().load(value)
                except (TypeError, ValidationError):
                    raise ValidationError(f"Invalid params for scheduler: {value}, see SearchAlgorithmConfig class.")
            raise ValidationError("Field should be dict")

        def _jsonschema_type_mapping(self):
            return {
                **schema_utils.unload_jsonschema_from_marshmallow_class(BaseSearchAlgorithmConfig),
                "title": "scheduler",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = lambda: BaseSearchAlgorithmConfig.Schema().load(default)
    dump_default = BaseSearchAlgorithmConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": SearchAlgorithmMarshmallowField(
                allow_none=False,
                load_default=load_default,
                dump_default=dump_default,
                metadata={"description": description, "parameter_metadata": None},
            )
        },
        default_factory=load_default,
    )
