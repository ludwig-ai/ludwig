from abc import ABC

from marshmallow_dataclass import dataclass
from ludwig.schema import utils as schema_utils


@dataclass
class BaseDecoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for decoders. Not meant to be used directly."""

    type: str
    "Name corresponding to a decoder."


@dataclass
class PassthroughDecoderConfig(BaseDecoderConfig):
    """PassthroughDecoderConfig is a dataclass that configures the parameters used for a passthrough decoder."""

    type: str = "passthrough"


@dataclass
class RegressorConfig(BaseDecoderConfig):
    """RegressorConfig is a dataclass that configures the parameters used for a regressor decoder"""

    type: str = "regressor"

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


@dataclass
class ProjectorConfig(BaseDecoderConfig):
    """ProjectorConfig is a dataclass that configures the parameters used for a projector decoder"""

    type: str = "projector"

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

    clip = None,


@dataclass
class ClassifierConfig(BaseDecoderConfig):

    type: str = "classifier"

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the input to the decoder.",
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of classes to predict.",
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

