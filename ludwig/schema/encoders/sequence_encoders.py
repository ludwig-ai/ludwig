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

