from typing import List

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig


@dataclass
class SequencePassthroughConfig(BaseEncoderConfig):

    type: str = "passthrough"

    reduce_output: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=256,
        description="The maximum length of a sequence.",
    )

    encoding_size: int = schema_utils.PositiveInteger(
        default=None,
        description="The size of the encoding vector, or None if sequence elements are scalars.",
    )


@dataclass
class SequenceEmbedConfig(BaseEncoderConfig):

    type: str = "embed"

    representation: str = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="Representation of the embedding.",
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the embedding.",
    )

    embeddings_trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether the embedding is trainable.",
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        description="Path to a file containing pretrained embeddings.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout probability for the embedding.",
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        default="uniform",
        description="Initializer to use for the weights matrix.",
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
    )


@dataclass
class ParallelCNNConfig(BaseEncoderConfig):

    type: str = "parallel_cnn"

    representation: str = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="Representation of the embedding.",
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the embedding.",
    )

    embeddings_trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether the embedding is trainable.",
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        description="Path to a file containing pretrained embeddings.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
    )

    conv_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for conv_layers
        default=None,
        description="List of dictionaries containing the parameters for each convolutional layer.",
    )

    num_conv_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of parallel convolutional layers to use.",
    )

    filter_size: int = schema_utils.PositiveInteger(
        default=3,
        description="Size of the 1d convolutional filter.",
    )

    num_filters: int = schema_utils.PositiveInteger(
        default=256,
        description="Number of filters, and by consequence number of output channels of the 1d convolution.",
    )

    pool_function: str = schema_utils.ReductionOptions(
        default="max",
        description="Pooling function to use.",
    )

    pool_size: int = schema_utils.PositiveInteger(
        default=None,
        description="The default pool_size that will be used for each layer.",
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of parallel fully connected layers to use.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
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

    reduce_output: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
    )


@dataclass
class StackedCNNConfig(BaseEncoderConfig):

    type: str = "stacked_cnn"

    representation: str = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="Representation of the embedding.",
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the embedding.",
    )

    embeddings_trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether the embedding is trainable.",
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        description="Path to a file containing pretrained embeddings.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
    )

    conv_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for conv_layers
        default=None,
        description="List of dictionaries containing the parameters for each convolutional layer.",
    )

    num_conv_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of parallel convolutional layers to use.",
    )

    filter_size: int = schema_utils.PositiveInteger(
        default=3,
        description="Size of the 1d convolutional filter.",
    )

    num_filters: int = schema_utils.PositiveInteger(
        default=256,
        description="Number of filters, and by consequence number of output channels of the 1d convolution.",
    )

    strides: int = schema_utils.PositiveInteger(
        default=1,
        description="Stride length of the convolution.",
    )

    padding: str = schema_utils.StringOptions(
        ["valid", "same"],
        default="same",
        description="Padding to use.",
    )

    dilation_rate: int = schema_utils.PositiveInteger(
        default=1,
        description="Dilation rate to use for dilated convolution.",
    )

    pool_function: str = schema_utils.ReductionOptions(
        default="max",
        description="Pooling function to use.",
    )

    pool_size: int = schema_utils.PositiveInteger(
        default=None,
        description="The default pool_size that will be used for each layer.",
    )

    pool_strides: int = schema_utils.PositiveInteger(
        default=None,
        description="Factor to scale down.",
    )

    pool_padding: str = schema_utils.StringOptions(
        ["valid", "same"],
        default="same",
        description="Padding to use.",
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of parallel fully connected layers to use.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
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

    reduce_output: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
    )


@dataclass
class StackedParallelCNNConfig(BaseEncoderConfig):

    type: str = "stacked_parallel_cnn"

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings",
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary of the input feature to encode",
    )

    representation: str = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="The representation of the embeddings. 'Dense' means the embeddings are initialized randomly. "
        "'Sparse' means they are initialized to be one-hot encodings.",
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The maximum embedding size. The actual size will be `min(vocabulary_size, embedding_size)` for "
        "`dense` representations and exactly `vocabulary_size` for the `sparse` encoding, "
        "where `vocabulary_size` is the number of different strings appearing in the training set in the "
        "column the feature is named after (plus 1 for `<UNK>`).",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None, description="The maximum length of all sequences"
    )

    embeddings_trainable: bool = schema_utils.Boolean(
        default=True,
        description="If true embeddings are trained during the training process, if false embeddings are fixed. It "
        "may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter "
        "has effect only when representation is dense as sparse one-hot encodings are not trainable. ",
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        description="By default dense embeddings are initialized randomly, but this parameter allows to specify a "
        "path to a file containing embeddings in the GloVe format. When the file containing the "
        "embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, "
        "the others are discarded. If the vocabulary contains strings that have no match in the "
        "embeddings file, their embeddings are initialized with the average of all other embedding plus "
        "some random noise to make them different from each other. This parameter has effect only if "
        "representation is dense.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="by default embedding matrices are stored on GPU memory if a GPU is used, as it allows for faster "
        "access, but in some cases the embedding matrix may be too large. This parameter forces the "
        "placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, "
        "slightly slowing down the process as a result of data transfer between CPU and GPU memory.",
    )

    stacked_layers: List[dict] = schema_utils.DictList(
        default=None,
        description="a nested list of lists of dictionaries containing the parameters of the stack of parallel "
        "convolutional layers. The length of the list determines the number of stacked parallel "
        "convolutional layers, length of the sub-lists determines the number of parallel conv layers and "
        "the content of each dictionary determines the parameters for a specific layer. ",
    )

    num_stacked_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="If stacked_layers is null, this is the number of elements in the stack of parallel convolutional "
        "layers. ",
    )

    filter_size: int = schema_utils.PositiveInteger(
        default=3,
        description="Size of the 1d convolutional filter.",
    )

    num_filters: int = schema_utils.PositiveInteger(
        default=256,
        description="Number of filters, and by consequence number of output channels of the 1d convolution.",
    )

    pool_function: str = schema_utils.ReductionOptions(
        default="max",
        description="Pooling function to use.",
    )

    pool_size: int = schema_utils.PositiveInteger(
        default=None,
        description="The default pool_size that will be used for each layer.",
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of parallel fully connected layers to use.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
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

    reduce_output: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
    )


@dataclass
class StackedRNNConfig(BaseEncoderConfig):

    type: str = "rnn"

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings",
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary of the input feature to encode",
    )

    representation: str = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="The representation of the embeddings. 'Dense' means the embeddings are initialized randomly. "
        "'Sparse' means they are initialized to be one-hot encodings.",
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The maximum embedding size. The actual size will be `min(vocabulary_size, embedding_size)` for "
        "`dense` representations and exactly `vocabulary_size` for the `sparse` encoding, "
        "where `vocabulary_size` is the number of different strings appearing in the training set in the "
        "column the feature is named after (plus 1 for `<UNK>`).",
    )
    embeddings_trainable: bool = schema_utils.Boolean(
        default=True,
        description="If true embeddings are trained during the training process, if false embeddings are fixed. It "
        "may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter "
        "has effect only when representation is dense as sparse one-hot encodings are not trainable. ",
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        description="By default dense embeddings are initialized randomly, but this parameter allows to specify a "
        "path to a file containing embeddings in the GloVe format. When the file containing the "
        "embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, "
        "the others are discarded. If the vocabulary contains strings that have no match in the "
        "embeddings file, their embeddings are initialized with the average of all other embedding plus "
        "some random noise to make them different from each other. This parameter has effect only if "
        "representation is dense.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="by default embedding matrices are stored on GPU memory if a GPU is used, as it allows for faster "
        "access, but in some cases the embedding matrix may be too large. This parameter forces the "
        "placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, "
        "slightly slowing down the process as a result of data transfer between CPU and GPU memory.",
    )

    num_layers: int = schema_utils.PositiveInteger(default=1, description="the number of stacked recurrent layers.")

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None, description="The maximum length of all sequences"
    )

    state_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The size of the state of the rnn.",
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

    unit_forget_bias: bool = schema_utils.Boolean(
        default=True,
        description="If true, add 1 to the bias of the forget gate at initialization",
    )

    recurrent_initializer: str = schema_utils.InitializerOptions(
        default="orthogonal", description="The initializer for recurrent matrix weights"
    )

    dropout: float = schema_utils.FloatRange(default=0.0, min=0, max=1, description="The dropout rate")

    recurrent_dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="The dropout rate for the recurrent state"
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="Number of parallel fully connected layers to use.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
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

    fc_activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each fully connected layer."
    )

    fc_dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="The dropout rate for fully connected layers"
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="last",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
    )


@dataclass
class StackedCNNRNNConfig(BaseEncoderConfig):

    type: str = "cnnrnn"

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings",
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary of the input feature to encode",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None, description="The maximum length of all sequences"
    )

    representation: str = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="The representation of the embeddings. 'Dense' means the embeddings are initialized randomly. "
        "'Sparse' means they are initialized to be one-hot encodings.",
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The maximum embedding size. The actual size will be `min(vocabulary_size, embedding_size)` for "
        "`dense` representations and exactly `vocabulary_size` for the `sparse` encoding, "
        "where `vocabulary_size` is the number of different strings appearing in the training set in the "
        "column the feature is named after (plus 1 for `<UNK>`).",
    )
    embeddings_trainable: bool = schema_utils.Boolean(
        default=True,
        description="If true embeddings are trained during the training process, if false embeddings are fixed. It "
        "may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter "
        "has effect only when representation is dense as sparse one-hot encodings are not trainable. ",
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        description="By default dense embeddings are initialized randomly, but this parameter allows to specify a "
        "path to a file containing embeddings in the GloVe format. When the file containing the "
        "embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, "
        "the others are discarded. If the vocabulary contains strings that have no match in the "
        "embeddings file, their embeddings are initialized with the average of all other embedding plus "
        "some random noise to make them different from each other. This parameter has effect only if "
        "representation is dense.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="By default embedding matrices are stored on GPU memory if a GPU is used, as it allows for faster "
        "access, but in some cases the embedding matrix may be too large. This parameter forces the "
        "placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, "
        "slightly slowing down the process as a result of data transfer between CPU and GPU memory.",
    )

    conv_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for conv_layers
        default=None,
        description="List of dictionaries containing the parameters for each convolutional layer.",
    )

    num_conv_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of parallel convolutional layers to use.",
    )

    num_filters: int = schema_utils.PositiveInteger(
        default=256,
        description="Number of filters, and by consequence number of output channels of the 1d convolution.",
    )

    filter_size: int = schema_utils.PositiveInteger(
        default=5,
        description="Size of the 1d convolutional filter.",
    )

    strides: int = schema_utils.PositiveInteger(
        default=1,
        description="Stride length of the convolution.",
    )

    padding: str = schema_utils.StringOptions(
        ["valid", "same"],
        default="same",
        description="Padding to use.",
    )

    dilation_rate: int = schema_utils.PositiveInteger(
        default=1,
        description="Dilation rate to use for dilated convolution.",
    )

    conv_activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each convolutional layer."
    )

    conv_dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="The dropout rate for the convolutional layers"
    )

    pool_function: str = schema_utils.ReductionOptions(
        default="max",
        description="Pooling function to use.",
    )

    pool_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The default pool_size that will be used for each layer.",
    )

    pool_strides: int = schema_utils.PositiveInteger(
        default=None,
        description="Factor to scale down.",
    )

    pool_padding: str = schema_utils.StringOptions(
        ["valid", "same"],
        default="same",
        description="Padding to use.",
    )

    num_rec_layers: int = schema_utils.PositiveInteger(default=1, description="The number of stacked recurrent layers.")

    state_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The size of the state of the rnn.",
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

    unit_forget_bias: bool = schema_utils.Boolean(
        default=True,
        description="If true, add 1 to the bias of the forget gate at initialization",
    )

    recurrent_initializer: str = schema_utils.InitializerOptions(
        default="orthogonal", description="The initializer for recurrent matrix weights"
    )

    dropout: float = schema_utils.FloatRange(default=0.0, min=0, max=1, description="The dropout rate")

    recurrent_dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="The dropout rate for the recurrent state"
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="Number of parallel fully connected layers to use.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
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

    fc_activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each fully connected layer."
    )

    fc_dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="The dropout rate for fully connected layers"
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="last",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
    )


@dataclass
class StackedTransformerConfig(BaseEncoderConfig):

    type: str = "transformer"

    max_sequence_length: int = schema_utils.PositiveInteger(default=None, description="Max length of all sequences")

    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings",
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary of the input feature to encode",
    )

    representation: str = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="The representation of the embeddings. 'Dense' means the embeddings are initialized randomly. "
        "'Sparse' means they are initialized to be one-hot encodings.",
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the embedding.",
    )

    embeddings_trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether the embedding is trainable.",
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        description="Path to a file containing pretrained embeddings.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
        "resolve them.",
    )

    num_layers: int = schema_utils.PositiveInteger(default=1, description="the number of stacked recurrent layers.")

    hidden_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The size of the hidden representation within the transformer block. It is usually the same as "
        "the embedding_size, but if the two values are different, a projection layer will be added before "
        "the first transformer block.",
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads in each transformer block.",
    )

    transformer_output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the fully connected layer after self attention in the transformer block. This is usually "
        "the same as hidden_size and embedding_size.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout rate for the transformer block",
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="Number of parallel fully connected layers to use.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
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

    fc_activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each fully connected layer."
    )

    fc_dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="The dropout rate for fully connected layers"
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="last",
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
        "tensor is greater than 2.",
    )
