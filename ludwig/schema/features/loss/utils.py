from dataclasses import Field
from typing import Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.loss import get_loss_classes, get_loss_cls


@DeveloperAPI
def get_loss_conds(feature_type: str):
    """Returns a JSON schema of conditionals to validate against loss types for specific feature types."""
    conds = []
    for loss in get_loss_classes(feature_type):
        loss_cls = get_loss_cls(feature_type, loss)
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(loss_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        loss_cond = schema_utils.create_cond(
            {"type": loss},
            other_props,
        )
        conds.append(loss_cond)
    return conds


@DeveloperAPI
def LossDataclassField(feature_type: str, default: str) -> Field:
    loss_registry = get_loss_classes(feature_type)

    class LossSelection(schema_utils.TypeSelection):
        def __init__(self):
            super().__init__(registry=loss_registry, default_value=default)

        def get_schema_from_registry(self, key: str) -> Type[schema_utils.BaseMarshmallowConfig]:
            return get_loss_cls(feature_type, key)

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(loss_registry.keys()), "default": default},
                },
                "title": "loss_options",
                "allOf": get_loss_conds(feature_type),
            }

    return LossSelection().get_default_field()
