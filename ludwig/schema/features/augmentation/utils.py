from dataclasses import field

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.augmentation.base import BaseAugmentationContainerConfig
from ludwig.utils.registry import Registry


@DeveloperAPI
@dataclass(repr=False)
class ImageAugmentationContainerConfig(BaseAugmentationContainerConfig):
    """Augmentation container for image features."""

    pass


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


def get_augmentation_cls(name: str):
    return get_augmentation_config_registry()[name]


@DeveloperAPI
def AugmentationContainerDataclassField(default=None, description=""):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify an augmentation
    config.

    Args:
        default: The default augmentation config to use.
        description: The description of the augmentation config.

    Returns: Initialized dataclass field that converts an untyped dict with params to an augmentation config.
    """

    class AugmentationContainerMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a list for a valid augmentation config from the
        augmentation_registry and creates a corresponding JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            augmentation_list = []
            for augmentation in value:
                augmentation_op = augmentation[TYPE]
                augmentation_cls = get_augmentation_cls(augmentation_op)
                pre = augmentation_cls()
                try:
                    augmentation_list.append(pre.Schema().load(augmentation))
                except (TypeError, ValidationError) as error:
                    raise ValidationError(
                        f"Invalid augmentation params: {value}, see `{pre}` definition. Error: {error}"
                    )
            return augmentation_list

        @staticmethod
        def _jsonschema_type_mapping():
            return get_augmentation_jsonschema()

    try:
        augmentation_container = ImageAugmentationContainerConfig
        load_default = augmentation_container.Schema().load({})
        dump_default = augmentation_container.Schema().dump({})

        return field(
            metadata={
                "marshmallow_field": AugmentationContainerMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(f"Unsupported augmentation type. See augmentation_registry. " f"Details: {e}")


@DeveloperAPI
def get_augmentation_jsonschema():
    """This function returns a JSON augmenation schema.

    Returns: JSON Schema
    """
    augmentation_types = sorted(list(get_augmentation_config_registry().keys()))
    schema = {
        "type": "array",
        "minItems": 0,
        "items": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": augmentation_types,
                    "title": "type",
                    "description": "Type of augmentation to apply.",
                },
            },
            "additionalProperties": True,
            "allOf": get_augmentation_conds(),
            "required": ["type"],
            "title": "augmentation",
        },
        # "uniqueItemProperties": ["name"],
    }

    return schema


@DeveloperAPI
def get_augmentation_conds():
    """This function returns a list of if-then JSON clauses for each augmentation type along with their properties
    and constraints.

    Returns: List of JSON clauses
    """
    # input_feature_types = sorted(list(input_config_registry.keys()))
    augmentation_types = sorted(list(get_augmentation_config_registry().keys()))
    conds = []
    # for feature_type in input_feature_types:  # TODO: placeholder for future use
    for augmentation_type in augmentation_types:
        schema_cls = get_augmentation_cls(augmentation_type)
        augmentation_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        augmentation_props = augmentation_schema["properties"]
        schema_utils.remove_duplicate_fields(augmentation_props)
        augmentation_cond = schema_utils.create_cond({"type": augmentation_type}, augmentation_props)
        conds.append(augmentation_cond)
    return conds
