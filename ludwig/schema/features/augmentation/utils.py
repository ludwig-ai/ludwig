from dataclasses import field
from typing import List, Union

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

_augmentation_config_registry = Registry()


@DeveloperAPI
def get_augmentation_config_registry() -> Registry:
    return _augmentation_config_registry


@DeveloperAPI
def register_augmentation_config(name: str, features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            augmentation_registry = get_augmentation_config_registry().get(feature, {})
            augmentation_registry[name] = cls
            get_augmentation_config_registry()[feature] = augmentation_registry
        return cls

    return wrap


@DeveloperAPI
def get_augmentation_cls(feature: str, name: str):
    return get_augmentation_config_registry()[feature][name]


@DeveloperAPI
def get_augmentation_classes(feature: str):
    return get_augmentation_config_registry()[feature]


@DeveloperAPI
def AugmentationContainerDataclassField(feature_type: str, default=[], description=""):
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
            assert isinstance(value, list), "Augmentation config must be a list."

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
            return get_augmentation_jsonschema(feature_type)

    try:
        if default:
            assert isinstance(default, list), "Augmentation config must be a list."
            augmentation_list = []
            for augmentation in default:
                augmentation_op = augmentation[TYPE]
                augmentation_cls = get_augmentation_cls(augmentation_op)
                pre = augmentation_cls()
                try:
                    augmentation_list.append(pre.Schema().load(augmentation))
                except (TypeError, ValidationError) as error:
                    raise ValidationError(
                        f"Invalid augmentation params: {default}, see `{pre}` definition. Error: {error}"
                    )
            load_default = dump_default = augmentation_list
        else:
            load_default = dump_default = default

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
def get_augmentation_jsonschema(feature_type: str):
    """This function returns a JSON augmenation schema.

    Returns: JSON Schema
    """
    augmentation_types = sorted(list(get_augmentation_config_registry()[feature_type].keys()))
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
            "allOf": get_augmentation_conds(feature_type),
            "required": ["type"],
            "title": "augmentation",
        },
        # "uniqueItemProperties": ["name"],
    }

    return schema


@DeveloperAPI
def get_augmentation_conds(feature_type: str):
    """This function returns a list of if-then JSON clauses for each augmentation type along with their properties
    and constraints.

    Returns: List of JSON clauses
    """
    conds = []
    for augmentation_op in get_augmentation_classes(feature_type):
        schema_cls = get_augmentation_cls(feature_type, augmentation_op)
        augmentation_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        augmentation_props = augmentation_schema["properties"]
        schema_utils.remove_duplicate_fields(augmentation_props)
        augmentation_cond = schema_utils.create_cond({"type": augmentation_op}, augmentation_props)
        conds.append(augmentation_cond)
    return conds
