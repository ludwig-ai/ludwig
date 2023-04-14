from collections import defaultdict

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MODEL_ECD, MODEL_GBM, MODEL_LLM
from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

input_config_registries = defaultdict(Registry)
output_config_registries = defaultdict(Registry)

ecd_input_config_registry = input_config_registries[MODEL_ECD]
gbm_input_config_registry = input_config_registries[MODEL_GBM]
llm_input_config_registry = input_config_registries[MODEL_LLM]

ecd_output_config_registry = output_config_registries[MODEL_ECD]
gbm_output_config_registry = output_config_registries[MODEL_GBM]
llm_output_config_registry = output_config_registries[MODEL_LLM]

input_mixin_registry = Registry()
output_mixin_registry = Registry()

"""
As of Ludwig v0.7, ECD models support the full range of feature parameters available in Ludwig, so any feature schema
can be registered into it. See `BinaryDefaultsConfig` for an example.
"""
ecd_defaults_config_registry = Registry()

"""
As of Ludwig v0.7, GBM models only support certain feature types and those features may only contain preprocessing
parameters (in comparison, ECD features can specify encoders and other parameters). This is why the two model types have
separate defaults registries. See `BinaryInputFeatureConfigMixin` for an example of a schema pattern that is designed to
be registered by this registry (whereas, conversely, `BinaryDefaultsConfig` is an example of one to be registered with
the ECD defaults registry).
"""
gbm_defaults_config_registry = Registry()

llm_defaults_config_registry = Registry()


def input_config_registry(model_type: str) -> Registry:
    return input_config_registries[model_type]


def output_config_registry(model_type: str) -> Registry:
    return output_config_registries[model_type]


@DeveloperAPI
def get_input_feature_cls(model_type: str, name: str):
    # TODO(travis): not needed once we remove existing model config implementation
    return input_config_registries[model_type][name]


@DeveloperAPI
def get_output_feature_cls(model_type: str, name: str):
    # TODO(ksbrar): What is this?
    return output_config_registries[model_type][name]


@DeveloperAPI
def get_input_feature_jsonschema(model_type: str):
    """This function returns a JSON schema structured to only requires a `type` key and then conditionally applies
    a corresponding input feature's field constraints.

    Returns: JSON Schema
    """
    input_feature_types = sorted(list(input_config_registry(model_type).keys()))
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "title": "name", "description": "Name of the input feature."},
            "type": {
                "type": "string",
                "enum": input_feature_types,
                "title": "type",
                "description": "Type of the input feature",
            },
            "column": {"type": "string", "title": "column", "description": "Name of the column."},
        },
        "additionalProperties": True,
        "allOf": get_input_feature_conds(model_type),
        "required": ["name", "type"],
        "title": "input_feature",
    }

    return schema


@DeveloperAPI
def get_input_feature_conds(model_type: str):
    """This function returns a list of if-then JSON clauses for each input feature type along with their properties
    and constraints.

    Returns: List of JSON clauses
    """
    input_feature_types = sorted(list(input_config_registry(model_type).keys()))
    conds = []
    for feature_type in input_feature_types:
        schema_cls = get_input_feature_cls(model_type, feature_type)
        feature_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        feature_props = feature_schema["properties"]
        schema_utils.remove_duplicate_fields(feature_props)

        feature_cond = schema_utils.create_cond({"type": feature_type}, feature_props)
        conds.append(feature_cond)
    return conds


@DeveloperAPI
def get_output_feature_jsonschema(model_type: str):
    """This function returns a JSON schema structured to only requires a `type` key and then conditionally applies
    a corresponding output feature's field constraints.

    Returns: JSON Schema
    """
    output_feature_types = sorted(list(output_config_registry(model_type).keys()))
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "title": "name", "description": "Name of the output feature."},
            "type": {
                "type": "string",
                "enum": output_feature_types,
                "title": "type",
                "description": "Type of the output feature",
            },
            "column": {"type": "string", "title": "column", "description": "Name of the column."},
        },
        "additionalProperties": True,
        "allOf": get_output_feature_conds(model_type),
        "required": ["name", "type"],
        "title": "output_feature",
    }

    return schema


@DeveloperAPI
def get_output_feature_conds(model_type: str):
    """This function returns a list of if-then JSON clauses for each output feature type along with their
    properties and constraints.

    Returns: List of JSON clauses
    """
    output_feature_types = sorted(list(output_config_registry(model_type).keys()))
    conds = []
    for feature_type in output_feature_types:
        schema_cls = get_output_feature_cls(model_type, feature_type)
        feature_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        feature_props = feature_schema["properties"]
        schema_utils.remove_duplicate_fields(feature_props)
        feature_cond = schema_utils.create_cond({"type": feature_type}, feature_props)
        conds.append(feature_cond)
    return conds
