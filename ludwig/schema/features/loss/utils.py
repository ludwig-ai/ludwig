from dataclasses import field

from marshmallow import fields, ValidationError

from ludwig.constants import TYPE
from ludwig.modules.loss_modules import get_loss_classes, get_loss_cls
from ludwig.schema import utils as schema_utils


def get_loss_conds(feature_type: str):
    """Returns a JSON schema of conditionals to validate against loss types for specific feature types."""
    conds = []
    for loss in get_loss_classes(feature_type):
        loss_cls = get_loss_cls(feature_type, loss).get_schema_cls()
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(loss_cls)["properties"]
        other_props.pop("type")
        loss_cond = schema_utils.create_cond(
            {"type": loss},
            other_props,
        )
        conds.append(loss_cond)
    return conds


def LossDataclassField(feature_type: str, default: str):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a loss config for
    the decoder of an output feature.

    Returns: Initialized dataclass field that converts an untyped dict with params to a loss config.
    """

    class LossMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict for a valid loss config from the
        preprocessing_registry and creates a corresponding JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if TYPE in value and value[TYPE] in get_loss_classes(feature_type):
                    loss_config = get_loss_cls(feature_type, value[TYPE]).get_schema_cls()
                    try:
                        return loss_config.Schema().load(value)
                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid loss params: {value}, see `{loss_config}` definition. Error: {error}"
                        )
                raise ValidationError(
                    f"Invalid params for loss: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            loss_classes = list(get_loss_classes(feature_type).keys())

            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": loss_classes, "default": default},
                },
                "title": "loss_options",
                "allOf": get_loss_conds(feature_type),
            }

    try:
        loss = get_loss_cls(feature_type, default).get_schema_cls()
        load_default = loss.Schema().load({"type": default})
        dump_default = loss.Schema().dump({"type": default})

        return field(
            metadata={
                "marshmallow_field": LossMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(f"Unsupported loss type: {default}. See loss_registry. " f"Details: {e}")
