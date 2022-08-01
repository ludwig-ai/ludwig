from abc import ABC
from typing import List, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils


@dataclass
class BaseEncoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for encoders.

    Not meant to be used directly.
    """

    type: str
    "Name corresponding to an encoder."


@dataclass
class PassthroughEncoderConfig(BaseEncoderConfig):
    """PassthroughEncoderConfig is a dataclass that configures the parameters used for a passthrough encoder."""

    type: str = "passthrough"


@dataclass
class DenseEncoderConfig(BaseEncoderConfig):
    """DenseEncoderConfig is a dataclass that configures the parameters used for a dense encoder."""

    type: str = "dense"

    fc_layers: List[dict] = schema_utils.DictList(
        default=None,
        description="List of fully connected layers to use in the encoder.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="Number of stacked fully connected layers that the input to the feature passes through.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the output of the feature.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: Union[str, dict] = schema_utils.StringOptions(  # TODO: Add support for String/Dict
        [
            "constant",
            "identity",
            "zeros",
            "ones",
            "orthogonal",
            "normal",
            "uniform",
            "truncated_normal",
            "variance_scaling",
            "glorot_normal",
            "glorot_uniform",
            "xavier_normal",
            "xavier_uniform",
            "he_normal",
            "he_uniform",
            "lecun_normal",
            "lecun_uniform",
        ],
        default="glorot_uniform",
        description="Initializer for the weight matrix.",
    )

    bias_initializer: Union[str, dict] = schema_utils.StringOptions(  # TODO: Add support for String/Dict
        [
            "constant",
            "identity",
            "zeros",
            "ones",
            "orthogonal",
            "normal",
            "uniform",
            "truncated_normal",
            "variance_scaling",
            "glorot_normal",
            "glorot_uniform",
            "xavier_normal",
            "xavier_uniform",
            "he_normal",
            "he_uniform",
            "lecun_normal",
            "lecun_uniform",
        ],
        default="zeros",
        description="Initializer for the bias vector.",
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

    activation: str = schema_utils.StringOptions(
        ["elu", "leakyRelu", "logSigmoid", "relu", "sigmoid", "tanh", "softmax"],
        default="relu",
        description="Activation function to apply to the output.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate.",
    )
