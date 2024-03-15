from abc import ABC
from dataclasses import field
from typing import Dict

from marshmallow import fields, ValidationError

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MODEL_ECD
from ludwig.schema.metadata import TRAINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class GradualUnfreezerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Configuration for gradual unfreezing parameters."""

    thaw_epochs: list = schema_utils.List(
        int,
        default=None,
        description="Epochs to thaw at. For example, [1, 2, 3, 4] will thaw layers in layers_to_thaw 2D array",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["gradual_unfreezer"]["thaw_epochs"],
    )

    layers_to_thaw: list = schema_utils.List(
        list,
        inner_type=str,
        default=None,
        description="Individual layers to thaw at each thaw_epoch. 2D Array",
        parameter_metadata=TRAINER_METADATA[MODEL_ECD]["gradual_unfreezer"]["layers_to_thaw"],
    )


@DeveloperAPI
def GradualUnfreezerDataclassField(description: str, default: Dict = None):
    allow_none = True
    default = default or {}

    class GradualUnfreezerMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return value
            if isinstance(value, dict):
                try:
                    return GradualUnfreezerConfig.Schema().load(value)
                except (TypeError, ValidationError) as e:
                    raise ValidationError(
                        f"Invalid params for gradual unfreezer: {value}, see GradualUnfreezerConfig class. Error: {e}"
                    )
            raise ValidationError("Field should be None or dict")

        def _jsonschema_type_mapping(self):
            return {
                **schema_utils.unload_jsonschema_from_marshmallow_class(GradualUnfreezerConfig),
                "title": "gradual_unfreeze_options",
                "description": description,
            }

    if not isinstance(default, dict):
        raise ValidationError(f"Invalid default: `{default}`")

    load_default = lambda: GradualUnfreezerConfig.Schema().load(default)
    dump_default = GradualUnfreezerConfig.Schema().dump(default)

    return field(
        metadata={
            "marshmallow_field": GradualUnfreezerMarshmallowField(
                allow_none=allow_none,
                load_default=load_default,
                dump_default=dump_default,
                metadata={
                    "description": description,
                },
            )
        },
        default_factory=load_default,
    )
