from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig


@dataclass
class SequenceGeneratorDecoderConfig(BaseDecoderConfig):

    type: str = "generator"

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the vocabulary.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the sequences.",
    )

    cell_type: str = schema_utils.StringOptions(
        ["rnn", "lstm", "gru"],
        default="gru",
        description="Type of recurrent cell to use.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the input to the decoder.",
    )

    reduce_input: str = schema_utils.StringOptions(
        ["sum", "mean", "avg", "max", "concat", "last"],
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked recurrent layers.",
    )


@dataclass
class SequenceTaggerDecoderConfig(BaseDecoderConfig):

    type: str = "tagger"

    input_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the input to the decoder.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the vocabulary.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the sequences.",
    )

    use_attention: bool = schema_utils.Boolean(
        default=False,
        description="Whether to apply a multi-head self attention layer before prediction.",
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    attention_embedding_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The embedding size of the multi-head self attention layer.",
    )

    attention_num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="The number of attention heads in the multi-head self attention layer.",
    )
