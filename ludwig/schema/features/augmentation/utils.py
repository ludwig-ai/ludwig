from dataclasses import field
from typing import List, Union

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.augmentation.base import BaseAugmentationConfig
from ludwig.utils.registry import Registry

# TODO: Is all of this needed?
# augmentation_registry = Registry()
#
#
# @DeveloperAPI
# def register_augmentation(name: str):
#     def wrap(augmentation_config: BaseAugmentationConfig):
#         augmentation_registry[name] = augmentation_config
#         return augmentation_config
#
#     return wrap

_augmentation_config_registry = Registry()


@DeveloperAPI
def get_augmentation_config_registry() -> Registry:
    return _augmentation_config_registry


def register_augmentation_config(name: str):
    def wrap(cls):
        get_augmentation_config_registry()[name] = cls
        return cls

    return wrap


def get_augmentation_op(feature: str, name: str):
    return get_augmentation_config_registry()[feature][name]


@DeveloperAPI
def AugmentationDataclassField():
    """Custom dataclass field that when used inside a dataclass will allow the user to specify an augmentation
    config.

    Returns: Initialized dataclass field that converts an untyped dict with params to an augmentation config.
    """

    class AugmentationMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a list for a valid augmentation config from the
        augmentation_registry and creates a corresponding JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            for augmentation in value:
                if isinstance(augmentation, str) and augmentation in get_augmentation_config_registry():
                    return augmentation
                elif isinstance(augmentation, dict):
                    pre = get_augmentation_config_registry()[tuple(augmentation.items())[0][0]]
                    try:
                        return pre.Schema().load(augmentation)
                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid augmentation params: {value}, see `{pre}` definition. Error: {error}"
                        )
                raise ValidationError(
                    f"Invalid params for augmentation: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            # augmentation_cls = _augmentation_config_registry[feature_type]
            props = schema_utils.unload_jsonschema_from_marshmallow_class(AugmentationMarshmallowField)["properties"]
            return {
                "type": "object",
                "properties": props,
                "title": "augmentation_options",
                "additionalProperties": True,
            }

    try:
        # augmentation_config = _augmentation_config_registry[feature_type]
        load_default = None  # augmentation_config.Schema().load({"feature_type": feature_type})
        dump_default = None  # augmentation_config.Schema().dump({"feature_type": feature_type})

        return field(
            metadata={
                "marshmallow_field": AugmentationMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(
            f"Unsupported augmentation type. See augmentation_registry. " f"Details: {e}"
        )
