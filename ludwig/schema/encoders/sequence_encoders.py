from dataclasses import Field
from typing import Any, Dict, List, Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUDIO, SEQUENCE, TEXT, TIMESERIES
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.schema.utils import ludwig_dataclass

CONV_LAYERS_DESCRIPTION = """
A list of dictionaries containing the parameters of all the convolutional layers.
The length of the list determines the number of stacked convolutional layers and the content of each dictionary
determines the parameters for a specific layer. The available parameters for each layer are: `activation`, `dropout`,
`norm`, `norm_params`, `num_filters`, `filter_size`, `strides`, `padding`, `dilation_rate`, `use_bias`, `pool_function`,
`pool_padding`, `pool_size`, `pool_strides`, `bias_initializer`, `weights_initializer`. If any of those values is
missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both
`conv_layers` and `num_conv_layers` are `null`, a default list will be assigned to `conv_layers` with the value
`[{filter_size: 7, pool_size: 3}, {filter_size: 7, pool_size: 3}, {filter_size: 3, pool_size: null},
{filter_size: 3, pool_size: null}, {filter_size: 3, pool_size: null}, {filter_size: 3, pool_size: 3}]`.
"""

NUM_CONV_LAYERS_DESCRIPTION = "The number of stacked convolutional layers when `conv_layers` is `null`."


def NumFiltersField(default: int = 256) -> Field:
    return schema_utils.PositiveInteger(
        default=default,
        description="Number of filters, and by consequence number of output channels of the 1d convolution.",
        parameter_metadata=ENCODER_METADATA["conv_params"]["num_filters"],
    )


def FilterSizeField(default: int = 3) -> Field:
    return schema_utils.PositiveInteger(
        default=default,
        description="Size of the 1d convolutional filter. It indicates how wide the 1d convolutional filter is.",
        parameter_metadata=ENCODER_METADATA["conv_params"]["filter_size"],
    )


def PoolFunctionField(default: str = "max") -> Field:
    return schema_utils.ReductionOptions(
        default=default,
        description=(
            "Pooling function to use. `max` will select the maximum value. Any of `average`, `avg`, or "
            "`mean` will compute the mean value"
        ),
        parameter_metadata=ENCODER_METADATA["conv_params"]["pool_function"],
    )


def PoolSizeField(default: Optional[int] = None) -> Field:
    return schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description=(
            "The default pool_size that will be used for each layer. If a pool_size is not already specified "
            "in conv_layers this is the default pool_size that will be used for each layer. It indicates the size of "
            "the max pooling that will be performed along the `s` sequence dimension after the convolution operation."
        ),
        parameter_metadata=ENCODER_METADATA["conv_params"]["pool_size"],
    )


@DeveloperAPI
@ludwig_dataclass
class SequenceEncoderConfig(BaseEncoderConfig):
    """Base class for sequence encoders."""

    def get_fixed_preprocessing_params(self) -> Dict[str, Any]:
        return {"cache_encoder_embeddings": False}


@DeveloperAPI
@register_encoder_config("passthrough", [TIMESERIES])
@ludwig_dataclass
class SequencePassthroughConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "SequencePassthrough"

    type: str = schema_utils.ProtectedString(
        "passthrough",
        description=ENCODER_METADATA["SequencePassthrough"]["type"].long_description,
    )

    max_sequence_length: int = common_fields.MaxSequenceLengthField(default=256)

    encoding_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The size of the encoding vector, or None if sequence elements are scalars.",
        parameter_metadata=ENCODER_METADATA["SequencePassthrough"]["encoding_size"],
    )

    reduce_output: str = common_fields.ReduceOutputField(default=None)


@DeveloperAPI
@register_encoder_config("embed", [SEQUENCE, TEXT])
@ludwig_dataclass
class SequenceEmbedConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "SequenceEmbed"

    type: str = schema_utils.ProtectedString(
        "embed",
        description=ENCODER_METADATA["SequenceEmbed"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField(description="Dropout rate applied to the embedding.")

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

    representation: str = common_fields.RepresentationField()

    vocab: list = common_fields.VocabField()

    weights_initializer: str = common_fields.WeightsInitializerField(default="uniform")

    reduce_output: str = common_fields.ReduceOutputField()

    embedding_size: int = common_fields.EmbeddingSizeField()

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()


@DeveloperAPI
@register_encoder_config("parallel_cnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
@ludwig_dataclass
class ParallelCNNConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "ParallelCNN"

    type: str = schema_utils.ProtectedString(
        "parallel_cnn",
        description=ENCODER_METADATA["ParallelCNN"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField(description="Dropout rate applied to the embedding.")

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer."
    )

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

    representation: str = common_fields.RepresentationField()

    vocab: list = common_fields.VocabField()

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
        parameter_metadata=ENCODER_METADATA["ParallelCNN"]["use_bias"],
    )

    bias_initializer: str = common_fields.BiasInitializerField()

    weights_initializer: str = common_fields.WeightsInitializerField()

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="Whether to embed the input sequence.",
        parameter_metadata=ENCODER_METADATA["ParallelCNN"]["should_embed"],
    )

    embedding_size: int = common_fields.EmbeddingSizeField()

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()

    reduce_output: str = common_fields.ReduceOutputField()

    num_conv_layers: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description=NUM_CONV_LAYERS_DESCRIPTION,
        parameter_metadata=ENCODER_METADATA["conv_params"]["num_conv_layers"],
    )

    conv_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for conv_layers
        default=None,
        description=CONV_LAYERS_DESCRIPTION,
        parameter_metadata=ENCODER_METADATA["conv_params"]["conv_layers"],
    )

    num_filters: int = NumFiltersField()

    filter_size: int = FilterSizeField()

    pool_function: str = PoolFunctionField()

    pool_size: int = PoolSizeField()

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["ParallelCNN"]["output_size"],
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="The default norm that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["ParallelCNN"]["norm"],
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
        parameter_metadata=ENCODER_METADATA["ParallelCNN"]["norm_params"],
    )

    num_fc_layers: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of parallel fully connected layers to use.",
        parameter_metadata=ENCODER_METADATA["ParallelCNN"]["num_fc_layers"],
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
        parameter_metadata=ENCODER_METADATA["ParallelCNN"]["fc_layers"],
    )


@DeveloperAPI
@register_encoder_config("stacked_cnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
@ludwig_dataclass
class StackedCNNConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "StackedCNN"

    type: str = schema_utils.ProtectedString(
        "stacked_cnn",
        description=ENCODER_METADATA["StackedCNN"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField(description="Dropout rate applied to the embedding.")

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer."
    )

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

    representation: str = common_fields.RepresentationField()

    vocab: list = common_fields.VocabField()

    num_conv_layers: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description=NUM_CONV_LAYERS_DESCRIPTION,
        parameter_metadata=ENCODER_METADATA["conv_params"]["num_conv_layers"],
    )

    conv_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for conv_layers
        default=None,
        description=CONV_LAYERS_DESCRIPTION,
        parameter_metadata=ENCODER_METADATA["conv_params"]["conv_layers"],
    )

    num_filters: int = NumFiltersField()

    filter_size: int = FilterSizeField()

    pool_function: str = PoolFunctionField()

    pool_size: int = PoolSizeField()

    strides: int = schema_utils.PositiveInteger(
        default=1,
        description="Stride length of the convolution.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["strides"],
    )

    padding: str = schema_utils.StringOptions(
        ["valid", "same"],
        default="same",
        description="Padding to use.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["padding"],
    )

    dilation_rate: int = schema_utils.PositiveInteger(
        default=1,
        description="Dilation rate to use for dilated convolution.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["dilation_rate"],
    )

    pool_strides: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Factor to scale down.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["pool_strides"],
    )

    pool_padding: str = schema_utils.StringOptions(
        ["valid", "same"],
        default="same",
        description="Padding to use.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["pool_padding"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["use_bias"],
    )

    bias_initializer: str = common_fields.BiasInitializerField()

    weights_initializer: str = common_fields.WeightsInitializerField()

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="Whether to embed the input sequence.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["should_embed"],
    )

    embedding_size: int = common_fields.EmbeddingSizeField()

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()

    reduce_output: str = common_fields.ReduceOutputField()

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["output_size"],
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="The default norm that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["norm"],
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["norm_params"],
    )

    num_fc_layers: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of parallel fully connected layers to use.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["num_fc_layers"],
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
        parameter_metadata=ENCODER_METADATA["StackedCNN"]["fc_layers"],
    )


@DeveloperAPI
@register_encoder_config("stacked_parallel_cnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
@ludwig_dataclass
class StackedParallelCNNConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "StackedParallelCNN"

    type: str = schema_utils.ProtectedString(
        "stacked_parallel_cnn",
        description=ENCODER_METADATA["StackedParallelCNN"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField(description="Dropout rate applied to the embedding.")

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer."
    )

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

    representation: str = common_fields.RepresentationField()

    vocab: list = common_fields.VocabField()

    num_stacked_layers: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="If stacked_layers is null, this is the number of elements in the stack of parallel convolutional "
        "layers. ",
        parameter_metadata=ENCODER_METADATA["StackedParallelCNN"]["num_stacked_layers"],
    )

    stacked_layers: List[dict] = schema_utils.DictList(
        default=None,
        description="a nested list of lists of dictionaries containing the parameters of the stack of parallel "
        "convolutional layers. The length of the list determines the number of stacked parallel "
        "convolutional layers, length of the sub-lists determines the number of parallel conv layers and "
        "the content of each dictionary determines the parameters for a specific layer. ",
        parameter_metadata=ENCODER_METADATA["StackedParallelCNN"]["stacked_layers"],
    )

    num_filters: int = NumFiltersField()

    filter_size: int = FilterSizeField()

    pool_function: str = PoolFunctionField()

    pool_size: int = PoolSizeField()

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
        parameter_metadata=ENCODER_METADATA["StackedParallelCNN"]["use_bias"],
    )

    bias_initializer: str = common_fields.BiasInitializerField()

    weights_initializer: str = common_fields.WeightsInitializerField()

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings",
        parameter_metadata=ENCODER_METADATA["StackedParallelCNN"]["should_embed"],
    )

    embedding_size: int = common_fields.EmbeddingSizeField()

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()

    reduce_output: str = common_fields.ReduceOutputField()

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["StackedParallelCNN"]["output_size"],
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="The default norm that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["StackedParallelCNN"]["norm"],
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
        parameter_metadata=ENCODER_METADATA["StackedParallelCNN"]["norm_params"],
    )

    num_fc_layers: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of parallel fully connected layers to use.",
        parameter_metadata=ENCODER_METADATA["StackedParallelCNN"]["num_fc_layers"],
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
        parameter_metadata=ENCODER_METADATA["StackedParallelCNN"]["fc_layers"],
    )


@DeveloperAPI
@register_encoder_config("rnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
@ludwig_dataclass
class StackedRNNConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "StackedRNN"

    type: str = schema_utils.ProtectedString(
        "rnn",
        description=ENCODER_METADATA["StackedRNN"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField(description="Dropout rate.")

    recurrent_dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout rate for the recurrent state",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["recurrent_dropout"],
    )

    activation: str = schema_utils.ActivationOptions(default="tanh", description="The default activation function.")

    recurrent_activation: str = schema_utils.ActivationOptions(
        default="sigmoid",
        description="The activation function to use in the recurrent step",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["recurrent_activation"],
    )

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

    representation: str = common_fields.RepresentationField()

    vocab: list = common_fields.VocabField()

    cell_type: str = schema_utils.StringOptions(
        ["rnn", "lstm", "gru"],
        default="rnn",
        description="The type of recurrent cell to use. Available values are: `rnn`, `lstm`, `gru`. For reference "
        "about the differences between the cells please refer to "
        "[torch.nn Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers).",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["cell_type"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked recurrent layers.",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["num_layers"],
    )

    state_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The size of the state of the rnn.",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["state_size"],
    )

    bidirectional: bool = schema_utils.Boolean(
        default=False,
        description="If true, two recurrent networks will perform encoding in the forward and backward direction and "
        "their outputs will be concatenated.",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["bidirectional"],
    )

    unit_forget_bias: bool = schema_utils.Boolean(
        default=True,
        description="If true, add 1 to the bias of the forget gate at initialization",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["unit_forget_bias"],
    )

    recurrent_initializer: str = schema_utils.InitializerOptions(
        default="orthogonal",
        description="The initializer for recurrent matrix weights",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["recurrent_initializer"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["use_bias"],
    )

    bias_initializer: str = common_fields.BiasInitializerField()

    weights_initializer: str = common_fields.WeightsInitializerField()

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["should_embed"],
    )

    embedding_size: int = common_fields.EmbeddingSizeField()

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()

    reduce_output: str = common_fields.ReduceOutputField(default="last")

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["StackedRNN"]["output_size"],
    )

    norm: str = common_fields.NormField(description="The default norm that will be used for each layer.")

    norm_params: dict = common_fields.NormParamsField()

    num_fc_layers: int = common_fields.NumFCLayersField(description="Number of parallel fully connected layers to use.")

    fc_activation: str = schema_utils.ActivationOptions()

    fc_dropout: float = common_fields.DropoutField()

    fc_layers: List[dict] = common_fields.FCLayersField()


@DeveloperAPI
@register_encoder_config("cnnrnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
@ludwig_dataclass
class StackedCNNRNNConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "StackedCNNRNN"

    type: str = schema_utils.ProtectedString(
        "cnnrnn",
        description=ENCODER_METADATA["StackedCNNRNN"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField(description="Dropout rate.")

    recurrent_dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout rate for the recurrent state",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["recurrent_dropout"],
    )

    conv_dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout rate for the convolutional layers",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["conv_dropout"],
    )

    activation: str = schema_utils.ActivationOptions(
        default="tanh", description="The default activation function to use."
    )

    recurrent_activation: str = schema_utils.ActivationOptions(
        default="sigmoid",
        description="The activation function to use in the recurrent step",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["recurrent_activation"],
    )

    conv_activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each convolutional layer.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["conv_activation"],
    )

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

    representation: str = common_fields.RepresentationField()

    vocab: list = common_fields.VocabField()

    cell_type: str = schema_utils.StringOptions(
        ["rnn", "lstm", "gru"],
        default="rnn",
        description="The type of recurrent cell to use. Available values are: `rnn`, `lstm`, `gru`. For reference "
        "about the differences between the cells please refer to "
        "[torch.nn Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers).",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["cell_type"],
    )

    num_conv_layers: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description=NUM_CONV_LAYERS_DESCRIPTION,
        parameter_metadata=ENCODER_METADATA["conv_params"]["num_conv_layers"],
    )

    conv_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for conv_layers
        default=None,
        description=CONV_LAYERS_DESCRIPTION,
        parameter_metadata=ENCODER_METADATA["conv_params"]["conv_layers"],
    )

    num_filters: int = NumFiltersField()

    filter_size: int = FilterSizeField(default=5)

    pool_function: str = PoolFunctionField()

    pool_size: int = PoolSizeField(default=2)

    strides: int = schema_utils.PositiveInteger(
        default=1,
        description="Stride length of the convolution.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["strides"],
    )

    padding: str = schema_utils.StringOptions(
        ["valid", "same"],
        default="same",
        description="Padding to use.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["padding"],
    )

    dilation_rate: int = schema_utils.PositiveInteger(
        default=1,
        description="Dilation rate to use for dilated convolution.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["dilation_rate"],
    )

    pool_strides: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Factor to scale down.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["pool_strides"],
    )

    pool_padding: str = schema_utils.StringOptions(
        ["valid", "same"],
        default="same",
        description="Padding to use.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["pool_padding"],
    )

    num_rec_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked recurrent layers.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["num_rec_layers"],
    )

    state_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The size of the state of the rnn.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["state_size"],
    )

    bidirectional: bool = schema_utils.Boolean(
        default=False,
        description="If true, two recurrent networks will perform encoding in the forward and backward direction and "
        "their outputs will be concatenated.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["bidirectional"],
    )

    unit_forget_bias: bool = schema_utils.Boolean(
        default=True,
        description="If true, add 1 to the bias of the forget gate at initialization",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["unit_forget_bias"],
    )

    recurrent_initializer: str = schema_utils.InitializerOptions(
        default="orthogonal",
        description="The initializer for recurrent matrix weights",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["recurrent_initializer"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["use_bias"],
    )

    bias_initializer: str = common_fields.BiasInitializerField()

    weights_initializer: str = common_fields.WeightsInitializerField()

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["should_embed"],
    )

    embedding_size: int = common_fields.EmbeddingSizeField()

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()

    reduce_output: str = common_fields.ReduceOutputField(default="last")

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["StackedCNNRNN"]["output_size"],
    )

    norm: str = common_fields.NormField(description="The default norm that will be used for each layer.")

    norm_params: dict = common_fields.NormParamsField()

    num_fc_layers: int = common_fields.NumFCLayersField(description="Number of parallel fully connected layers to use.")

    fc_activation: str = schema_utils.ActivationOptions()

    fc_dropout: float = common_fields.DropoutField()

    fc_layers: List[dict] = common_fields.FCLayersField()


@DeveloperAPI
@register_encoder_config("transformer", [SEQUENCE, TEXT, TIMESERIES])
@ludwig_dataclass
class StackedTransformerConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "StackedTransformer"

    type: str = schema_utils.ProtectedString(
        "transformer",
        description=ENCODER_METADATA["StackedTransformer"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField(default=0.1, description="The dropout rate for the transformer block.")

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

    representation: str = common_fields.RepresentationField()

    vocab: list = common_fields.VocabField()

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of transformer layers.",
        parameter_metadata=ENCODER_METADATA["StackedTransformer"]["num_layers"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The size of the hidden representation within the transformer block. It is usually the same as "
        "the embedding_size, but if the two values are different, a projection layer will be added before "
        "the first transformer block.",
        parameter_metadata=ENCODER_METADATA["StackedTransformer"]["hidden_size"],
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads in each transformer block.",
        parameter_metadata=ENCODER_METADATA["StackedTransformer"]["num_heads"],
    )

    transformer_output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the fully connected layer after self attention in the transformer block. This is usually "
        "the same as hidden_size and embedding_size.",
        parameter_metadata=ENCODER_METADATA["StackedTransformer"]["transformer_output_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
        parameter_metadata=ENCODER_METADATA["StackedTransformer"]["use_bias"],
    )

    bias_initializer: str = common_fields.BiasInitializerField()

    weights_initializer: str = common_fields.WeightsInitializerField()

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings",
        parameter_metadata=ENCODER_METADATA["StackedTransformer"]["should_embed"],
    )

    embedding_size: int = common_fields.EmbeddingSizeField()

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()

    reduce_output: str = common_fields.ReduceOutputField(default="last")

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["StackedTransformer"]["output_size"],
    )

    norm: str = common_fields.NormField(description="The default norm that will be used for each layer.")

    norm_params: dict = common_fields.NormParamsField()

    num_fc_layers: int = common_fields.NumFCLayersField(description="Number of parallel fully connected layers to use.")

    fc_activation: str = schema_utils.ActivationOptions()

    fc_dropout: float = common_fields.DropoutField()

    fc_layers: List[dict] = common_fields.FCLayersField()
