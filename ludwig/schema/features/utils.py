from dataclasses import field
from marshmallow import fields, ValidationError

from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry
from ludwig.schema.features.base import BasePreprocessingConfig

input_type_registry = Registry()
output_type_registry = Registry()
preprocessing_registry = Registry()


def register_input_feature(name: str):
    def wrap(cls):
        input_type_registry[name] = cls
        return cls

    return wrap


def register_output_feature(name: str):
    def wrap(cls):
        output_type_registry[name] = cls
        return cls

    return wrap


def register_preprocessor(name: str):
    def wrap(preprocessing_config: BasePreprocessingConfig):
        preprocessing_registry[name] = preprocessing_config
        return preprocessing_config

    return wrap


def get_input_feature_jsonschema():
    """This function returns a JSON schema structured to only requires a `type` key and then conditionally applies
    a corresponding input feature's field constraints.

    Returns: JSON Schema
    """
    input_feature_types = sorted(list(input_type_registry.keys()))
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": input_feature_types},
                "column": {"type": "string"},
            },
            "additionalProperties": True,
            "allOf": get_input_feature_conds(),
            "required": ["name", "type"],
        },
    }


def get_input_feature_conds():
    """This function returns a list of if-then JSON clauses for each input feature type along with their properties
    and constraints.

    Returns: List of JSON clauses
    """
    input_feature_types = sorted(list(input_type_registry.keys()))
    conds = []
    for feature_type in input_feature_types:
        feature_cls = input_type_registry[feature_type]
        schema_cls = feature_cls.get_schema_cls()
        feature_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        feature_props = feature_schema["properties"]
        feature_cond = schema_utils.create_cond({"type": feature_type}, feature_props)
        conds.append(feature_cond)
    return conds


def get_output_feature_jsonschema():
    """This function returns a JSON schema structured to only requires a `type` key and then conditionally applies
    a corresponding output feature's field constraints.

    Returns: JSON Schema
    """
    output_feature_types = sorted(list(output_type_registry.keys()))
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": output_feature_types},
                "column": {"type": "string"},
            },
            "additionalProperties": True,
            "allOf": get_output_feature_conds(),
            "required": ["name", "type"],
        },
    }


def get_output_feature_conds():
    """This function returns a list of if-then JSON clauses for each output feature type along with their
    properties and constraints.

    Returns: List of JSON clauses
    """
    output_feature_types = sorted(list(output_type_registry.keys()))
    conds = []
    for feature_type in output_feature_types:
        feature_cls = output_type_registry[feature_type]
        schema_cls = feature_cls.get_schema_cls()
        feature_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        feature_props = feature_schema["properties"]
        feature_cond = schema_utils.create_cond({"type": feature_type}, feature_props)
        conds.append(feature_cond)
    return conds


def PreprocessingDataclassField(feature_type: str):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a preprocessing
    config.

    Returns: Initialized dataclass field that converts an untyped dict with params to a preprocessing config.
    """

    class PreprocessingMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict for a valid preprocessing config from the
        preprocessing_registry and creates a corresponding JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if feature_type in preprocessing_registry:
                    pre = preprocessing_registry[feature_type]
                    try:
                        return pre.Schema().load(value)
                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid preprocessing params: {value}, see `{pre}` definition. Error: {error}"
                        )
                raise ValidationError(
                    f"Invalid params for preprocessor: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            preprocessor_cls = preprocessing_registry[feature_type]
            props = schema_utils.unload_jsonschema_from_marshmallow_class(preprocessor_cls)["properties"]
            return {
                "type": "object",
                "properties": props,
                "additionalProperties": False,
            }

    try:
        preprocessor = preprocessing_registry[feature_type]
        load_default = preprocessor.Schema().load({"feature_type": feature_type})
        dump_default = preprocessor.Schema().dump({"feature_type": feature_type})

        return field(
            metadata={
                "marshmallow_field": PreprocessingMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(
            f"Unsupported preprocessing type: {feature_type}. See preprocessing_registry. " f"Details: {e}"
        )
