from typing import Optional, Union, List, ClassVar
from ludwig.encoders.base import Encoder
from ludwig.encoders.sequence_encoders import (
    SequenceEmbedEncoder,
    ParallelCNN,
    StackedCNN,
    StackedParallelCNN,
    StackedRNN,
    StackedCNNRNN,
    StackedTransformer,

)

from marshmallow_dataclass import dataclass
from ludwig.schema import utils as schema_utils


@dataclass
class EmbedEncoderConfig(schema_utils.BaseMarshmallowConfig):

    encoder_class: ClassVar[Encoder] = SequenceEmbedEncoder

    type: str = "embed"

    representation: Optional[str] = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="Representation of the embedding.",
    )

    embedding_size: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        description="Size of the embedding.",
    )

    embeddings_trainable: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether the embedding is trainable.",
    )

    pretrained_embeddings: Optional[str] = schema_utils.String(
        default=None,
        description="Path to a file containing pretrained embeddings.",
    )

    embeddings_on_cpu: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
                    "resolve them.",
    )

    dropout: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description="Dropout probability for the embedding.",
    )

    weights_initializer: Optional[Union[str, dict]] = schema_utils.StringOptions(  # TODO: Add support for String/Dict
        ["constant", "identity", "zeros", "ones", "orthogonal", "normal", "uniform", "truncated_normal",
            "variance_scaling", "glorot_normal", "glorot_uniform", "xavier_normal", "xavier_uniform", "he_normal",
            "he_uniform", "lecun_normal", "lecun_uniform"],
        default="glorot_uniform",
        description="Initializer to use.",
    )

    reduce_output: Optional[str] = schema_utils.StringOptions(
        ["sum", "mean", "avg", "max", "concat", "last"],
        default="sum",
        allow_none=True,
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
                    "tensor is greater than 2.",
    )


@dataclass
class ParallelCNNConfig(schema_utils.BaseMarshmallowConfig):

    encoder_class: ClassVar[Encoder] = ParallelCNN

    type: str = "parallel_cnn"

    representation: Optional[str] = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="Representation of the embedding.",
    )

    embedding_size: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        description="Size of the embedding.",
    )

    embeddings_trainable: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether the embedding is trainable.",
    )

    pretrained_embeddings: Optional[str] = schema_utils.String(
        default=None,
        description="Path to a file containing pretrained embeddings.",
    )

    embeddings_on_cpu: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether to force the placement of the embedding matrix in regular memory and have the CPU "
                    "resolve them.",
    )

    conv_layers: Optional[List[dict]] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for conv_layers
        default=None,
        description="List of dictionaries containing the parameters for each convolutional layer.",
    )

    num_conv_layers: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        description="Number of parallel convolutional layers to use.",
    )

    filter_size: Optional[int] = schema_utils.PositiveInteger(
        default=3,
        description="Size of the 1d convolutional filter.",
    )

    num_filters: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        description="Number of filters, and by consequence number of output channels of the 1d convolution.",
    )

    pool_function: Optional[str] = schema_utils.StringOptions(
        ["max", "average", "avg", "mean"],
        default="max",
        description="Pooling function to use.",
    )

    pool_size: Optional[int] = schema_utils.PositiveInteger(
        default=2,
        description="Size of the pooling used in each layer.",
    )

    fc_layers: Optional[List[dict]] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
    )

    num_fc_layers: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        description="Number of parallel fully connected layers to use.",
    )

    output_size: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        description="The default output_size that will be used for each layer.",
    )

    use_bias: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether to use a bias vector.",
    )

    weights_initializer: Optional[Union[str, dict]] = schema_utils.StringOptions(  # TODO: Add support for String/Dict
        ["constant", "identity", "zeros", "ones", "orthogonal", "normal", "uniform", "truncated_normal",
         "variance_scaling", "glorot_normal", "glorot_uniform", "xavier_normal", "xavier_uniform", "he_normal",
         "he_uniform", "lecun_normal", "lecun_uniform"],
        default="glorot_uniform",
        description="Initializer to use.",
    )

    bias_initializer: Optional[Union[str, dict]] = schema_utils.StringOptions(  # TODO: Add support for String/Dict
        ["constant", "identity", "zeros", "ones", "orthogonal", "normal", "uniform", "truncated_normal",
            "variance_scaling", "glorot_normal", "glorot_uniform", "xavier_normal", "xavier_uniform", "he_normal",
            "he_uniform", "lecun_normal", "lecun_uniform"],
        default="zeros",
        description="Initializer to use.",
    )

    norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        description="The default norm that will be used for each layer.",
    )

    norm_params: Optional[dict] = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
    )

    activation: Optional[str] = schema_utils.String(
        default="relu",
        description="The default activation function that will be used for each layer.",
    )

    dropout: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description="Dropout probability for the embedding.",
    )

    reduce_output: Optional[str] = schema_utils.StringOptions(
        ["sum", "mean", "avg", "max", "concat", "last"],
        default="sum",
        allow_none=True,
        description="How to reduce the output tensor along the `s` sequence length dimension if the rank of the "
                    "tensor is greater than 2.",
    )


@dataclass
class StackedCNNConfig(schema_utils.BaseMarshmallowConfig):

    encoder_class: ClassVar[Encoder] = StackedCNN

    type: str = "stacked_cnn"


@dataclass
class StackedParallelCNNConfig(schema_utils.BaseMarshmallowConfig):

    encoder_class: ClassVar[Encoder] = StackedParallelCNN

    type: str = "stacked_parallel_cnn"


@dataclass
class StackedRNNConfig(schema_utils.BaseMarshmallowConfig):

    encoder_class: ClassVar[Encoder] = StackedRNN

    type: str = "rnn"


@dataclass
class StackedCNNRNNConfig(schema_utils.BaseMarshmallowConfig):

    encoder_class: ClassVar[Encoder] = StackedCNNRNN

    type: str = "cnnrnn"


@dataclass
class StackedTransformerConfig(schema_utils.BaseMarshmallowConfig):

    encoder_class: ClassVar[Encoder] = StackedTransformer

    type: str = "transformer"

