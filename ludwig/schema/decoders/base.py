from abc import ABC
from typing import Any, Dict, List, Tuple, Union

from marshmallow_dataclass import dataclass

from ludwig.constants import BINARY, CATEGORY, NUMBER, SEQUENCE, SET, TEXT, VECTOR
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.utils import register_decoder_config


@dataclass
class BaseDecoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for decoders.

    Not meant to be used directly.
    """

    type: str
    "Name corresponding to a decoder."

    fc_layers: List[Dict[str, Any]] = schema_utils.DictList(
        default=None, description="List of dictionaries containing the parameters for each fully connected layer."
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0, description="Number of fully-connected layers if fc_layers not specified."
    )

    fc_output_size: int = schema_utils.PositiveInteger(default=256, description="Output size of fully connected stack.")

    fc_use_bias: bool = schema_utils.Boolean(
        default=True, description="Whether the layer uses a bias vector in the fc_stack."
    )

    fc_weights_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="xavier_uniform", description="The weights initializer to use for the layers in the fc_stack"
    )

    fc_bias_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="zeros", description="The bias initializer to use for the layers in the fc_stack"
    )

    fc_norm: str = schema_utils.StringOptions(
        ["batch", "layer"], description="The normalization to use for the layers in the fc_stack"
    )

    fc_norm_params: dict = schema_utils.Dict(
        description="The additional parameters for the normalization in the fc_stack"
    )

    fc_activation: str = schema_utils.ActivationOptions(
        default="relu", description="The activation to use for the layers in the fc_stack"
    )

    fc_dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="The dropout rate to use for the layers in the fc_stack"
    )


@register_decoder_config("passthrough", [BINARY, CATEGORY, NUMBER, SET, VECTOR, SEQUENCE, TEXT])
@dataclass
class PassthroughDecoderConfig(BaseDecoderConfig):
    """PassthroughDecoderConfig is a dataclass that configures the parameters used for a passthrough decoder."""

    type: str = "passthrough"

    input_size: int = schema_utils.PositiveInteger(
        default=1,
        description="Size of the input to the decoder.",
    )


@register_decoder_config("regressor", [BINARY, NUMBER])
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


@register_decoder_config("projector", [VECTOR])
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

    clip: Union[List[int], Tuple[int]] = schema_utils.FloatRangeTupleDataclassField(
        n=2,
        default=None,
        allow_none=True,
        min=0,
        max=999999999,
        description="Clip the output of the decoder to be within the given range.",
    )


@register_decoder_config("classifier", [CATEGORY, SET])
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
