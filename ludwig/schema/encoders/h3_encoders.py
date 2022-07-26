from typing import List

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig


@dataclass
class H3EmbedConfig(BaseEncoderConfig):

    type: str = "embed"

    embedding_size: int = schema_utils.PositiveInteger(
        default=10,
        description="The maximum embedding size adopted.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of stacked fully connected layers.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=10,
        description="If an output_size is not already specified in fc_layers this is the default output_size that "
        "will be used for each layer. It indicates the size of the output of a fully connected layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer to use for the weights matrix.",
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer to use for the bias vector.",
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

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer."
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout probability for the embedding.",
    )


@dataclass
class H3WeightedSumConfig(BaseEncoderConfig):

    type: str = "embed"

    embedding_size: int = schema_utils.PositiveInteger(
        default=10,
        description="The maximum embedding size adopted.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
    )

    should_softmax: bool = schema_utils.Boolean(
        default=False,
        description="Determines if the weights of the weighted sum should be passed though a softmax layer before "
        "being used.",
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of stacked fully connected layers.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=10,
        description="If an output_size is not already specified in fc_layers this is the default output_size that "
        "will be used for each layer. It indicates the size of the output of a fully connected layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer to use for the weights matrix.",
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer to use for the bias vector.",
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

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer."
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout probability for the embedding.",
    )


@dataclass
class H3RNNConfig(BaseEncoderConfig):

    type: str = "rnn"

    embedding_size: int = schema_utils.PositiveInteger(
        default=10,
        description="The maximum embedding size adopted.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked recurrent layers.",
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=10,
        description="The size of the hidden representation within the transformer block. It is usually the same as "
        "the embedding_size, but if the two values are different, a projection layer will be added before "
        "the first transformer block.",
    )

    cell_type: str = schema_utils.StringOptions(
        ["rnn", "lstm", "lstm_block", "lstm", "ln", "lstm_cudnn", "gru", "gru_block", "gru_cudnn"],
        default="rnn",
        description="The type of recurrent cell to use. Available values are: `rnn`, `lstm`, `lstm_block`, `lstm`, "
        "`ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`. For reference about the differences between "
        "the cells please refer to TensorFlow's documentation. We suggest to use the `block` variants on "
        "CPU and the `cudnn` variants on GPU because of their increased speed. ",
    )

    bidirectional: bool = schema_utils.Boolean(
        default=False,
        description="If true, two recurrent networks will perform encoding in the forward and backward direction and "
        "their outputs will be concatenated.",
    )

    activation: str = schema_utils.ActivationOptions(
        default="tanh",
        description="The activation function to use",
    )

    recurrent_activation: str = schema_utils.ActivationOptions(
        default="sigmoid",
        description="The activation function to use in the recurrent step",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
    )

    unit_forget_bias: bool = schema_utils.Boolean(
        default=True,
        description="If true, add 1 to the bias of the forget gate at initialization",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer to use for the weights matrix.",
    )

    recurrent_initializer: str = schema_utils.InitializerOptions(
        default="orthogonal", description="The initializer for recurrent matrix weights"
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer to use for the bias vector.",
    )

    dropout: float = schema_utils.FloatRange(default=0.0, min=0, max=1, description="The dropout rate")

    recurrent_dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="The dropout rate for the recurrent state"
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="last",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
    )
