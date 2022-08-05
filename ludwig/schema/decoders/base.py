from abc import ABC
from typing import List

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils


@dataclass
class BaseDecoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for decoders.

    Not meant to be used directly.
    """

    type: str
    "Name corresponding to a decoder."

    fc_layers: List[dict] = None

    num_fc_layers: int = 0

    output_size: int = 256

    use_bias: bool = True

    weights_initializer: str = "xavier_uniform"

    bias_initializer: str = "zeros"

    norm: str = None

    norm_params: dict = None

    activation: str = "relu"

    dropout: float = 0


@dataclass
class PassthroughDecoderConfig(BaseDecoderConfig):
    """PassthroughDecoderConfig is a dataclass that configures the parameters used for a passthrough decoder."""

    type: str = "passthrough"


@dataclass
class RegressorConfig(BaseDecoderConfig):
    """RegressorConfig is a dataclass that configures the parameters used for a regressor decoder."""

    type: str = schema_utils.StringOptions(
        ["regressor"],
        default="regressor",
        allow_none=False,
        description="Type of decoder.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the decoder.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
    )

    threshold: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description="Threshold for the output of the decoder.",
    )


@dataclass
class ProjectorConfig(BaseDecoderConfig):
    """ProjectorConfig is a dataclass that configures the parameters used for a projector decoder."""

    type: str = schema_utils.StringOptions(
        ["projector"],
        default="projector",
        allow_none=False,
        description="Type of decoder.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the decoder.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the output of the decoder.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
    )

    activation: str = schema_utils.ActivationOptions(
        default=None,
        description=" Indicates the activation function applied to the output.",
    )

    clip = (None,)


@dataclass
class ClassifierConfig(BaseDecoderConfig):

    type: str = schema_utils.StringOptions(
        ["classifier"],
        default="classifier",
        allow_none=False,
        description="Type of decoder.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the decoder.",
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of classes to predict.",
    )

    fc_layers: List[dict] = schema_utils.DictList(
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
    )

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer."
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="The default norm that will be used for each layer.",
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
    )

    threshold: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description="The threshold above (greater or equal) which the predicted output of the sigmoid will be mapped "
        "to 1.",
    )
