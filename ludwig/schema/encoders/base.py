from abc import ABC
from typing import List, Union

from marshmallow_dataclass import dataclass

from ludwig.constants import BINARY, NUMBER, VECTOR
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.utils import register_encoder_config


@dataclass(repr=False, order=True)
class BaseEncoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for encoders."""

    type: str


@register_encoder_config("passthrough", [NUMBER, VECTOR])
@dataclass(order=True)
class PassthroughEncoderConfig(BaseEncoderConfig):
    """PassthroughEncoderConfig is a dataclass that configures the parameters used for a passthrough encoder."""

    type: str = schema_utils.StringOptions(
        ["passthrough"],
        default="passthrough",
        allow_none=False,
        description="Type of encoder.",
    )


@register_encoder_config("dense", [BINARY, NUMBER, VECTOR])
@dataclass(repr=False, order=True)
class DenseEncoderConfig(BaseEncoderConfig):
    """DenseEncoderConfig is a dataclass that configures the parameters used for a dense encoder."""

    type: str = schema_utils.StringOptions(
        ["dense"],
        default="dense",
        allow_none=False,
        description="Type of encoder.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate.",
    )

    activation: str = schema_utils.StringOptions(
        ["elu", "leakyRelu", "logSigmoid", "relu", "sigmoid", "tanh", "softmax"],
        default="relu",
        description="Activation function to apply to the output.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the dense encoder.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the output of the feature.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    bias_initializer: Union[str, dict] = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
    )

    weights_initializer: Union[str, dict] = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
    )

    norm: Union[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        allow_none=True,
        default=None,
        description="Normalization to use in the dense layer.",
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters for normalization if norm is either batch or layer.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="Number of stacked fully connected layers that the input to the feature passes through.",
    )

    fc_layers: List[dict] = schema_utils.DictList(
        default=None,
        description="List of fully connected layers to use in the encoder.",
    )
