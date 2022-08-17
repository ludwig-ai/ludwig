from marshmallow_dataclass import dataclass
from marshmallow import fields, ValidationError
from dataclasses import field

from typing import Union, List

from ludwig.constants import (
    BINARY_WEIGHTED_CROSS_ENTROPY,
    MEAN_SQUARED_ERROR,
    MEAN_ABSOLUTE_ERROR,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
    SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    SIGMOID_CROSS_ENTROPY,
    SOFTMAX_CROSS_ENTROPY,
    TYPE,
)

from ludwig.modules.loss_modules import get_loss_cls, get_loss_classes
from ludwig.schema import utils as schema_utils


@dataclass
class BaseLossConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for feature configs."""

    type: str

    weight: float


@dataclass
class MSELossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[MEAN_SQUARED_ERROR],
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class MAELossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[MEAN_ABSOLUTE_ERROR],
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class RMSELossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[ROOT_MEAN_SQUARED_ERROR],
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class RMSPELossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[ROOT_MEAN_SQUARED_PERCENTAGE_ERROR],
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class BWCEWLossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[BINARY_WEIGHTED_CROSS_ENTROPY],
        description="Type of loss.",
    )

    positive_class_weight: int = schema_utils.NonNegativeInteger(
        default=None,
        description="Weight of the positive class.",
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(
        default=0,
        description=""
    )

    confidence_penalty: int = schema_utils.NonNegativeInteger(
        default=0,
        description=""
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class SoftmaxCrossEntropyLossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[SOFTMAX_CROSS_ENTROPY],
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.OneOfOptionsField(
        default=1,
        allow_none=True,
        field_options=[
            schema_utils.List(list_type=float, default=None),
            schema_utils.NonNegativeFloat(default=1.0),
        ],
        description="Weights to apply to each class in the loss. If not specified, all classes are weighted equally.",
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(
        default=0,
        description=""
    )

    confidence_penalty: int = schema_utils.NonNegativeInteger(
        default=0,
        description=""
    )

    class_similarities_temperature: int = schema_utils.NonNegativeInteger(
        default=0,
        description=""
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


@dataclass
class SequenceSoftmaxCrossEntropyLossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[SEQUENCE_SOFTMAX_CROSS_ENTROPY],
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.OneOfOptionsField(
        default=1,
        allow_none=True,
        field_options=[
            schema_utils.List(list_type=float, default=None),
            schema_utils.NonNegativeFloat(default=1.0),
        ],
        description="Weights to apply to each class in the loss. If not specified, all classes are weighted equally.",
    )

    robust_lambda: int = schema_utils.NonNegativeInteger(
        default=0,
        description=""
    )

    confidence_penalty: int = schema_utils.NonNegativeInteger(
        default=0,
        description=""
    )

    class_similarities_temperature: int = schema_utils.NonNegativeInteger(
        default=0,
        description=""
    )
    
    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )
    
    unique: bool = schema_utils.Boolean(
        default=False,
        description="If true, the loss is only computed for unique elements in the sequence.",
    )


@dataclass
class SigmoidCrossEntropyLossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[SIGMOID_CROSS_ENTROPY],
        description="Type of loss.",
    )

    class_weights: Union[List[float], float, None] = schema_utils.OneOfOptionsField(
        default=1,
        allow_none=True,
        field_options=[
            schema_utils.List(list_type=float, default=None),
            schema_utils.NonNegativeFloat(default=1.0),
        ],
        description="Weights to apply to each class in the loss. If not specified, all classes are weighted equally.",
    )
    
    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )


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
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a loss
    config for the decoder of an output feature.

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
                "required": ["type"],
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
