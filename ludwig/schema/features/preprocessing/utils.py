from dataclasses import field

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.utils.registry import Registry

preprocessing_registry = Registry()


@DeveloperAPI
def register_preprocessor(name: str):
    def wrap(preprocessing_config: BasePreprocessingConfig):
        preprocessing_registry[name] = preprocessing_config
        return preprocessing_config

    return wrap


@DeveloperAPI
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
                "title": "preprocessing_options",
                "additionalProperties": True,
            }

    try:
        preprocessor = preprocessing_registry[feature_type]
        load_default = lambda: preprocessor.Schema().load({"feature_type": feature_type})
        dump_default = preprocessor.Schema().dump({"feature_type": feature_type})

        return field(
            metadata={
                "marshmallow_field": PreprocessingMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=load_default,
        )
    except Exception as e:
        raise ValidationError(
            f"Unsupported preprocessing type: {feature_type}. See preprocessing_registry. " f"Details: {e}"
        )
