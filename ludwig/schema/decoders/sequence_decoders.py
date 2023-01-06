from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import SEQUENCE, TEXT
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.metadata import DECODER_METADATA


@DeveloperAPI
@register_decoder_config("generator", [SEQUENCE, TEXT])
@dataclass(repr=False)
class SequenceGeneratorDecoderConfig(BaseDecoderConfig):
    type: str = schema_utils.ProtectedString(
        "generator",
        description="The generator decoder is a (potentially empty) stack of fully connected layers, followed by an "
        "RNN that generates outputs feeding on its own previous predictions and generates a tensor of "
        "size `b x s' x c`, where `b` is the batch size, `s'` is the length of the generated sequence and "
        "`c` is the number of classes, followed by a softmax_cross_entropy. During training teacher "
        "forcing is adopted, meaning the list of targets is provided as both inputs and outputs (shifted "
        "by 1), while at evaluation time greedy decoding (generating one token at a time and feeding it "
        "as input for the next step) is performed by beam search, using a beam of 1 by default. In "
        "general a generator expects a `b x h` shaped input tensor, where `h` is a hidden dimension. The "
        "`h` vectors are (after an optional stack of fully connected layers) fed into the rnn generator. "
        "One exception is when the generator uses attention, as in that case the expected size of the "
        "input tensor is `b x s x h`, which is the output of a sequence, text or time series input "
        "feature without reduced outputs or the output of a sequence-based combiner. If a `b x h` input "
        "is provided to a generator decoder using an RNN with attention instead, an error will be raised "
        "during model building.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the vocabulary.",
        parameter_metadata=DECODER_METADATA["SequenceGeneratorDecoder"]["vocab_size"],
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the sequences.",
        parameter_metadata=DECODER_METADATA["SequenceGeneratorDecoder"]["max_sequence_length"],
    )

    cell_type: str = schema_utils.StringOptions(
        ["rnn", "lstm", "gru"],
        default="gru",
        description="Type of recurrent cell to use.",
        parameter_metadata=DECODER_METADATA["SequenceGeneratorDecoder"]["cell_type"],
    )

    input_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["SequenceGeneratorDecoder"]["input_size"],
    )

    reduce_input: str = schema_utils.StringOptions(
        ["sum", "mean", "avg", "max", "concat", "last"],
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=DECODER_METADATA["SequenceGeneratorDecoder"]["reduce_input"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of stacked recurrent layers.",
        parameter_metadata=DECODER_METADATA["SequenceGeneratorDecoder"]["num_layers"],
    )


@DeveloperAPI
@register_decoder_config("tagger", [SEQUENCE, TEXT])
@dataclass(repr=False)
class SequenceTaggerDecoderConfig(BaseDecoderConfig):
    type: str = schema_utils.ProtectedString(
        "tagger",
        description="The tagger decoder is a (potentially empty) stack of fully connected layers, "
        "followed by a projection into a tensor of size `b x s x c`, where `b` is the batch size, "
        "`s` is the length of the sequence and `c` is the number of classes, followed by a "
        "softmax_cross_entropy. This decoder requires its input to be shaped as `b x s x h`, where `h` is "
        "a hidden dimension, which is the output of a sequence, text or time series input feature without "
        "reduced outputs or the output of a sequence-based combiner. If a `b x h` input is provided "
        "instead, an error will be raised during model building.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["SequenceTaggerDecoder"]["input_size"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Size of the vocabulary.",
        parameter_metadata=DECODER_METADATA["SequenceTaggerDecoder"]["vocab_size"],
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the sequences.",
        parameter_metadata=DECODER_METADATA["SequenceTaggerDecoder"]["max_sequence_length"],
    )

    use_attention: bool = schema_utils.Boolean(
        default=False,
        description="Whether to apply a multi-head self attention layer before prediction.",
        parameter_metadata=DECODER_METADATA["SequenceTaggerDecoder"]["use_attention"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=DECODER_METADATA["SequenceTaggerDecoder"]["use_bias"],
    )

    attention_embedding_size: int = schema_utils.PositiveInteger(
        default=256,
        description="The embedding size of the multi-head self attention layer.",
        parameter_metadata=DECODER_METADATA["SequenceTaggerDecoder"]["attention_embedding_size"],
    )

    attention_num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="The number of attention heads in the multi-head self attention layer.",
        parameter_metadata=DECODER_METADATA["SequenceTaggerDecoder"]["attention_num_heads"],
    )
