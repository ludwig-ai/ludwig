from abc import ABC
from typing import Any, Dict, List, Tuple, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, CATEGORY, NUMBER, SEQUENCE, SET, TEXT, VECTOR
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.initializers import (
    BiasInitializerDataclassField,
    InitializerConfig,
    WeightsInitializerDataclassField,
)
from ludwig.schema.metadata import DECODER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BaseDecoderConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for decoders."""

    type: str = schema_utils.StringOptions(
        ["regressor", "classifier", "projector", "generator", "tagger"],
        default=None,
        allow_none=True,
        description="The type of decoder to use.",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["type"],
    )

    fc_layers: List[Dict[str, Any]] = schema_utils.DictList(
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_layers"],
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="Number of fully-connected layers if fc_layers not specified.",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["num_fc_layers"],
    )

    fc_output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of fully connected stack.",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_output_size"],
    )

    fc_use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector in the fc_stack.",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_use_bias"],
    )

    fc_weights_initializer: InitializerConfig = WeightsInitializerDataclassField(
        default="xavier_uniform",
        description="The weights initializer to use for the layers in the fc_stack",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_weights_initializer"],
    )

    fc_bias_initializer: InitializerConfig = BiasInitializerDataclassField(
        default="zeros",
        description="The bias initializer to use for the layers in the fc_stack",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_bias_initializer"],
    )

    fc_norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="The normalization to use for the layers in the fc_stack",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_norm"],
    )

    fc_norm_params: dict = schema_utils.Dict(
        description="The additional parameters for the normalization in the fc_stack",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_norm_params"],
    )

    fc_activation: str = schema_utils.ActivationOptions(
        default="relu",
        description="The activation to use for the layers in the fc_stack",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_activation"],
    )

    fc_dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout rate to use for the layers in the fc_stack",
        parameter_metadata=DECODER_METADATA["BaseDecoder"]["fc_dropout"],
    )


@DeveloperAPI
@register_decoder_config("passthrough", [BINARY, CATEGORY, NUMBER, SET, VECTOR, SEQUENCE, TEXT])
@ludwig_dataclass
class PassthroughDecoderConfig(BaseDecoderConfig):
    """PassthroughDecoderConfig is a dataclass that configures the parameters used for a passthrough decoder."""

    @classmethod
    def module_name(cls):
        return "PassthroughDecoder"

    type: str = schema_utils.ProtectedString(
        "passthrough",
        description="The passthrough decoder simply returns the raw numerical values coming from the combiner as "
        "outputs",
        parameter_metadata=DECODER_METADATA["PassthroughDecoder"]["type"],
    )

    input_size: int = schema_utils.PositiveInteger(
        default=1,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["PassthroughDecoder"]["input_size"],
    )


@DeveloperAPI
@register_decoder_config("regressor", [BINARY, NUMBER])
@ludwig_dataclass
class RegressorConfig(BaseDecoderConfig):
    """RegressorConfig is a dataclass that configures the parameters used for a regressor decoder."""

    @classmethod
    def module_name(cls):
        return "Regressor"

    type: str = schema_utils.ProtectedString(
        "regressor",
        description=DECODER_METADATA["Regressor"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["Regressor"]["input_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["Regressor"]["use_bias"],
    )

    weights_initializer: InitializerConfig = WeightsInitializerDataclassField(
        description="Initializer for the weight matrix.",
        parameter_metadata=DECODER_METADATA["Regressor"]["weights_initializer"],
    )

    bias_initializer: InitializerConfig = BiasInitializerDataclassField(
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=DECODER_METADATA["Regressor"]["bias_initializer"],
    )


@DeveloperAPI
@register_decoder_config("projector", [VECTOR])
@ludwig_dataclass
class ProjectorConfig(BaseDecoderConfig):
    """ProjectorConfig is a dataclass that configures the parameters used for a projector decoder."""

    @classmethod
    def module_name(cls):
        return "Projector"

    type: str = schema_utils.ProtectedString(
        "projector",
        description=DECODER_METADATA["Projector"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["Projector"]["input_size"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the output of the decoder.",
        parameter_metadata=DECODER_METADATA["Projector"]["output_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["Projector"]["use_bias"],
    )

    weights_initializer: InitializerConfig = WeightsInitializerDataclassField(
        description="Initializer for the weight matrix.",
        parameter_metadata=DECODER_METADATA["Projector"]["weights_initializer"],
    )

    bias_initializer: InitializerConfig = BiasInitializerDataclassField(
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


@DeveloperAPI
@register_decoder_config("classifier", [CATEGORY, SET])
@ludwig_dataclass
class ClassifierConfig(BaseDecoderConfig):
    @classmethod
    def module_name(cls):
        return "Classifier"

    type: str = schema_utils.ProtectedString(
        "classifier",
        description=DECODER_METADATA["Classifier"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["Classifier"]["input_size"],
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of classes to predict.",
        parameter_metadata=DECODER_METADATA["Classifier"]["num_classes"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["Classifier"]["use_bias"],
    )

    weights_initializer: InitializerConfig = WeightsInitializerDataclassField(
        description="Initializer for the weight matrix.",
        parameter_metadata=DECODER_METADATA["Classifier"]["weights_initializer"],
    )

    bias_initializer: InitializerConfig = BiasInitializerDataclassField(
        default="zeros",
        description="Initializer for the bias vector.",
        parameter_metadata=DECODER_METADATA["Classifier"]["bias_initializer"],
    )
