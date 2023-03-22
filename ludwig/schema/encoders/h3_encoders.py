from typing import List

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import H3
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_encoder_config("embed", H3)
@ludwig_dataclass
class H3EmbedConfig(BaseEncoderConfig):
    @staticmethod
    def module_name():
        return "H3Embed"

    type: str = schema_utils.ProtectedString(
        "embed",
        description=ENCODER_METADATA["H3Embed"]["type"].long_description,
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout probability for the embedding.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["dropout"],
    )

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["activation"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["use_bias"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer to use for the bias vector.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["bias_initializer"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer to use for the weights matrix.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["weights_initializer"],
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=10,
        description="The maximum embedding size adopted.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["embedding_size"],
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["embeddings_on_cpu"],
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["reduce_output"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=10,
        description="If an output_size is not already specified in fc_layers this is the default output_size that "
        "will be used for each layer. It indicates the size of the output of a fully connected layer.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["output_size"],
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="The default norm that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["norm"],
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["norm_params"],
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of stacked fully connected layers.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["num_fc_layers"],
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
        parameter_metadata=ENCODER_METADATA["H3Embed"]["fc_layers"],
    )


@DeveloperAPI
@register_encoder_config("weighted_sum", H3)
@ludwig_dataclass
class H3WeightedSumConfig(BaseEncoderConfig):
    @staticmethod
    def module_name():
        return "H3WeightedSum"

    type: str = schema_utils.ProtectedString(
        "weighted_sum",
        description=ENCODER_METADATA["H3WeightedSum"]["type"].long_description,
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout probability for the embedding.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["dropout"],
    )

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["activation"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["use_bias"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer to use for the bias vector.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["bias_initializer"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer to use for the weights matrix.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["weights_initializer"],
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=10,
        description="The maximum embedding size adopted.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["embedding_size"],
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["embeddings_on_cpu"],
    )

    should_softmax: bool = schema_utils.Boolean(
        default=False,
        description="Determines if the weights of the weighted sum should be passed though a softmax layer before "
        "being used.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["should_softmax"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=10,
        description="If an output_size is not already specified in fc_layers this is the default output_size that "
        "will be used for each layer. It indicates the size of the output of a fully connected layer.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["output_size"],
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="The default norm that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["norm"],
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["norm_params"],
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of stacked fully connected layers.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["num_fc_layers"],
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
        parameter_metadata=ENCODER_METADATA["H3WeightedSum"]["fc_layers"],
    )


@DeveloperAPI
@register_encoder_config("rnn", H3)
@ludwig_dataclass
class H3RNNConfig(BaseEncoderConfig):
    @staticmethod
    def module_name():
        return "H3RNN"

    type: str = schema_utils.ProtectedString(
        "rnn",
        description=ENCODER_METADATA["H3RNN"]["type"].long_description,
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout rate",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["dropout"],
    )

    recurrent_dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout rate for the recurrent state",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["recurrent_dropout"],
    )

    activation: str = schema_utils.ActivationOptions(
        default="tanh",
        description="The activation function to use",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["activation"],
    )

    recurrent_activation: str = schema_utils.ActivationOptions(
        default="sigmoid",
        description="The activation function to use in the recurrent step",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["recurrent_activation"],
    )

    cell_type: str = schema_utils.StringOptions(
        ["rnn", "lstm", "lstm_block", "ln", "lstm_cudnn", "gru", "gru_block", "gru_cudnn"],
        default="rnn",
        description="The type of recurrent cell to use. Available values are: `rnn`, `lstm`, `lstm_block`, `lstm`, "
        "`ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`. For reference about the differences between "
        "the cells please refer to PyTorch's documentation. We suggest to use the `block` variants on "
        "CPU and the `cudnn` variants on GPU because of their increased speed. ",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["cell_type"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked recurrent layers.",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["num_layers"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=10,
        description="The size of the hidden representation within the transformer block. It is usually the same as "
        "the embedding_size, but if the two values are different, a projection layer will be added before "
        "the first transformer block.",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["hidden_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["use_bias"],
    )

    unit_forget_bias: bool = schema_utils.Boolean(
        default=True,
        description="If true, add 1 to the bias of the forget gate at initialization",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["unit_forget_bias"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer to use for the bias vector.",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["bias_initializer"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer to use for the weights matrix.",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["weights_initializer"],
    )

    recurrent_initializer: str = schema_utils.InitializerOptions(
        default="orthogonal",
        description="The initializer for recurrent matrix weights",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["recurrent_initializer"],
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="last",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["reduce_output"],
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=10,
        description="The maximum embedding size adopted.",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["embedding_size"],
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["embeddings_on_cpu"],
    )

    bidirectional: bool = schema_utils.Boolean(
        default=False,
        description="If true, two recurrent networks will perform encoding in the forward and backward direction and "
        "their outputs will be concatenated.",
        parameter_metadata=ENCODER_METADATA["H3RNN"]["bidirectional"],
    )
