from abc import ABC
from typing import Optional, Union, List, ClassVar
from ludwig.encoders.base import Encoder
from ludwig.encoders.generic_encoders import DenseEncoder, PassthroughEncoder

from marshmallow import Schema
from marshmallow_dataclass import dataclass
from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

encoder_registry = Registry()


def register_encoder(name: str):
    def wrap(encoder_config: BaseEncoderConfig):
        encoder_registry[name] = encoder_config
        return encoder_config

    return wrap


@dataclass
class BaseEncoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for encoders. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding Encoder class are copied over.
    """

    encoder_class: ClassVar[Optional[Encoder]] = None
    "Class variable pointing to the corresponding Encoder class."

    encoder: str
    "Name corresponding to an encoder."


@register_encoder(name="sgd")
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
        min_value=0.0,
        max_value=1.0,
        description="Dropout rate.",
    )


@register_encoder(name="passthrough")
class PassthroughEncoder(Schema):

    encoder_class: ClassVar[Encoder] = PassthroughEncoder

    type: str = "passthrough"
        