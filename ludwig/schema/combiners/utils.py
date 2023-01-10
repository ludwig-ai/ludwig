from typing import Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json
from ludwig.utils.registry import Registry

DEFAULT_VALUE = "concat"
DESCRIPTION = "Select the combiner type."

combiner_registry = Registry()


@DeveloperAPI
def register_combiner(name: str):
    def wrap(cls):
        combiner_registry[name] = cls
        return cls

    return wrap


@DeveloperAPI
def get_combiner_jsonschema():
    """Returns a JSON schema structured to only require a `type` key and then conditionally apply a corresponding
    combiner's field constraints."""
    combiner_types = sorted(list(combiner_registry.keys()))
    parameter_metadata = convert_metadata_to_json(COMBINER_METADATA[TYPE])
    return {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": combiner_types,
                "default": DEFAULT_VALUE,
                "title": "combiner_options",
                "description": DESCRIPTION,
                "parameter_metadata": parameter_metadata,
            },
        },
        "allOf": get_combiner_conds(),
        "required": ["type"],
    }


@DeveloperAPI
def get_combiner_conds():
    """Returns a list of if-then JSON clauses for each combiner type in `combiner_registry` and its properties'
    constraints."""
    combiner_types = sorted(list(combiner_registry.keys()))
    conds = []
    for combiner_type in combiner_types:
        combiner_cls = combiner_registry[combiner_type]
        schema_cls = combiner_cls.get_schema_cls()
        combiner_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        combiner_props = combiner_schema["properties"]
        schema_utils.remove_duplicate_fields(combiner_props)
        combiner_cond = schema_utils.create_cond({"type": combiner_type}, combiner_props)
        conds.append(combiner_cond)
    return conds


class CombinerSelection(schema_utils.TypeSelection):
    def __init__(self):
        # For registration of all combiners
        import ludwig.combiners.combiners  # noqa

        super().__init__(registry=combiner_registry, default_value=DEFAULT_VALUE, description=DESCRIPTION)

    def get_schema_from_registry(self, key: str) -> Type[schema_utils.BaseMarshmallowConfig]:
        return self.registry[key].get_schema_cls()

    @staticmethod
    def _jsonschema_type_mapping():
        return get_combiner_jsonschema()
