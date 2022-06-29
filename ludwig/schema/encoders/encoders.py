from abc import ABC
from dataclasses import field
from typing import Optional, Union, List, ClassVar
from ludwig.encoders.base import Encoder
from ludwig.encoders.generic_encoders import DenseEncoder, PassthroughEncoder
from ludwig.encoders.registry import get_encoder_classes, get_encoder_cls

from marshmallow import Schema, fields, ValidationError
from marshmallow_dataclass import dataclass
from ludwig.schema import utils as schema_utils


@dataclass
class BaseEncoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for encoders. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding Encoder class are copied over.
    """

    encoder_class: ClassVar[Optional[Encoder]] = None
    "Class variable pointing to the corresponding Encoder class."

    type: str
    "Name corresponding to an encoder."


@dataclass
class DenseEncoder(Schema):
    """DenseEncoder is a dataclass that configures the parameters used for a dense encoder."""

    encoder_class: ClassVar[Encoder] = DenseEncoder

    type: str = "dense"

    fc_layers: Optional[List[dict]] = schema_utils.DictList(
        default=None,
        description="List of fully connected layers to use in the encoder.",
    )

    num_layers: Optional[int] = schema_utils.PositiveInteger(
        default=1,
        description="Number of stacked fully connected layers that the input to the feature passes through.",
    )

    output_size: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        description="Size of the output of the feature.",
    )

    use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: Optional[Union[str, dict]] = schema_utils.StringOptions(  # TODO: Add support for String/Dict
        ["constant", "identity", "zeros", "ones", "orthogonal", "normal", "uniform", "truncated_normal",
         "variance_scaling", "glorot_normal", "glorot_uniform", "xavier_normal", "xavier_uniform", "he_normal",
         "he_uniform", "lecun_normal", "lecun_uniform"],
        default="glorot_uniform",
        description="Initializer for the weight matrix.",
    )

    bias_initializer: Optional[Union[str, dict]] = schema_utils.StringOptions(  # TODO: Add support for String/Dict
        ["constant", "identity", "zeros", "ones", "orthogonal", "normal", "uniform", "truncated_normal",
         "variance_scaling", "glorot_normal", "glorot_uniform", "xavier_normal", "xavier_uniform", "he_normal",
         "he_uniform", "lecun_normal", "lecun_uniform"],
        default="zeros",
        description="Initializer for the bias vector.",
    )

    norm: Optional[Union[str]] = schema_utils.StringOptions(
        ["batch", "layer"],
        allow_none=True,
        default=None,
        description="Normalization to use in the dense layer.",
    )

    norm_params: Optional[dict] = schema_utils.Dict(
        default=None,
        description="Parameters for normalization if norm is either batch or layer.",
    )

    activation: Optional[str] = schema_utils.StringOptions(
        ["elu", "leakyRelu", "logSigmoid", "relu", "sigmoid", "tanh", "softmax"],
        default="relu",
        description="Activation function to apply to the output.",
    )

    dropout: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description="Dropout rate.",
    )


@dataclass
class PassthroughEncoder(Schema):

    encoder_class: ClassVar[Encoder] = PassthroughEncoder

    type: str = "passthrough"


def EncoderDataclassField(feature_type: str, encoder_type: str):
    """
    Custom dataclass field that when used inside a dataclass will allow the user to specify a preprocessing config.

    Returns: Initialized dataclass field that converts an untyped dict with params to a preprocessing config.
    """

    class PreprocessingMarshmallowField(fields.Field):
        """
        Custom marshmallow field that deserializes a dict for a valid preprocessing config from the
        preprocessing_registry and creates a corresponding `oneOf` JSON schema for external usage.
        """

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if encoder_type in get_encoder_classes(feature_type):
                    encoder = get_encoder_cls(feature_type, encoder_type)
                    try:
                        return encoder.Schema().load(value)
                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid preprocessing params: {value}, see `{encoder}` definition. Error: {error}"
                        )
                raise ValidationError(
                    f"Invalid params for preprocessor: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            encoder_cls = get_encoder_cls(feature_type, encoder_type)
            props = schema_utils.unload_jsonschema_from_marshmallow_class(encoder_cls)["properties"]
            return {
                "type": "object",
                "properties": props,
                "additionalProperties": False,
            }

    try:
        encoder = get_encoder_cls(feature_type, encoder_type)
        load_default = encoder.Schema().load({'type': encoder_type})
        dump_default = encoder.Schema().dump({'type': encoder_type})

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
        raise ValidationError(f"Unsupported preprocessing type: {encoder_type}. See preprocessing_registry. "
                              f"Details: {e}")

