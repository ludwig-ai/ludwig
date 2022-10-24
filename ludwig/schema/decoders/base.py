from abc import ABC
from typing import Any, Dict, List, Tuple, Union

from marshmallow_dataclass import dataclass

from ludwig.constants import BINARY, CATEGORY, NUMBER, SEQUENCE, SET, TEXT, VECTOR
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.metadata.decoder_metadata import DECODER_METADATA


@dataclass(repr=False)
class BaseDecoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for decoders."""

    type: str

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
@dataclass(repr=False)
class PassthroughDecoderConfig(BaseDecoderConfig):
    """PassthroughDecoderConfig is a dataclass that configures the parameters used for a passthrough decoder."""

    type: str = "passthrough"

    input_size: int = schema_utils.PositiveInteger(
        default=1,
        description="Size of the input to the decoder.",
    )


@register_decoder_config("regressor", [BINARY, NUMBER])
@dataclass(repr=False)
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
        parameter_metadata=DECODER_METADATA["Regressor"]["input_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["Regressor"]["use_bias"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
        parameter_metadata=DECODER_METADATA["Regressor"]["weights_initializer"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=DECODER_METADATA["Regressor"]["bias_initializer"],
    )


@register_decoder_config("projector", [VECTOR])
@dataclass(repr=False)
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
        parameter_metadata=DECODER_METADATA["Projector"]["input_size"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the output of the decoder.",
        parameter_metadata=DECODER_METADATA["Projector"]["output_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["Projector"]["use_bias"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
        parameter_metadata=DECODER_METADATA["Projector"]["weights_initializer"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=DECODER_METADATA["Projector"]["bias_initializer"],
    )

    activation: str = schema_utils.ActivationOptions(
        default=None,
        description=" Indicates the activation function applied to the output.",
        parameter_metadata=DECODER_METADATA["Projector"]["activation"],
    )

    clip: Union[List[int], Tuple[int]] = schema_utils.FloatRangeTupleDataclassField(
        n=2,
        default=None,
        allow_none=True,
        min=0,
        max=999999999,
        description="Clip the output of the decoder to be within the given range.",
        parameter_metadata=DECODER_METADATA["Projector"]["clip"],
    )


@register_decoder_config("classifier", [CATEGORY, SET])
@dataclass(repr=False)
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
        parameter_metadata=DECODER_METADATA["Classifier"]["input_size"],
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of classes to predict.",
        parameter_metadata=DECODER_METADATA["Classifier"]["num_classes"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["Classifier"]["use_bias"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer for the weight matrix.",
        parameter_metadata=DECODER_METADATA["Classifier"]["weights_initializer"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=DECODER_METADATA["Classifier"]["bias_initializer"],
    )
