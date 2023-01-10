from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata
from ludwig.utils.registry import Registry

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
                "enum_descriptions": get_combiner_descriptions(),
                "default": "concat",
                "title": "combiner_options",
                "description": "Select the combiner type.",
                "parameter_metadata": parameter_metadata,
            },
        },
        "allOf": get_combiner_conds(),
        "required": ["type"],
    }


@DeveloperAPI
def get_combiner_descriptions():
    """
    This function returns a dictionary of combiner descriptions available at the type selection.
    The process works as follows - 1) Get a dictionary of valid combiners from the combiner config registry,
    but inverse the key/value pairs since we need to index `valid_combiners` later with an altered version
    of the combiner config class name. 2) Loop through Combiner Metadata entries, if a metadata entry has a
    combiner name that matches a valid combiner, add the description metadata to the output dictionary.

    :return: A dictionary of combiner descriptions
    """
    output = {}
    combiners = {
        class_name.__name__.replace("Config", ""): registered_name
        for registered_name, class_name
        in combiner_registry.items()
    }

    for k, v in COMBINER_METADATA.items():
        if any([k == name for name in combiners]) and not isinstance(v, ParameterMetadata):
            output[combiners[k]] = convert_metadata_to_json(v[TYPE])

    return output


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
