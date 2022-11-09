from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.combiner_metadata import COMBINER_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json
from ludwig.utils.registry import Registry

combiner_registry = Registry()


def register_combiner(name: str):
    def wrap(cls):
        combiner_registry[name] = cls
        return cls

    return wrap


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
                "default": "concat",
                "title": "combiner_options",
                "description": "Select the combiner type.",
                "parameter_metadata": parameter_metadata,
            },
        },
        "allOf": get_combiner_conds(),
        "required": ["type"],
    }


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
