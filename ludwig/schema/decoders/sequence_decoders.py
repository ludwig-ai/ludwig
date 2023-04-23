from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MODEL_ECD, SEQUENCE, TEXT
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.metadata import DECODER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_decoder_config("generator", [SEQUENCE, TEXT], model_types=[MODEL_ECD])
@ludwig_dataclass
class SequenceGeneratorDecoderConfig(BaseDecoderConfig):
    @staticmethod
    def module_name():
        return "SequenceGeneratorDecoder"

    type: str = schema_utils.ProtectedString(
        "generator",
        description=DECODER_METADATA["SequenceGeneratorDecoder"]["type"].long_description,
    )

    vocab_size: int = common_fields.VocabSizeField()

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

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
@register_decoder_config("tagger", [SEQUENCE, TEXT], model_types=[MODEL_ECD])
@ludwig_dataclass
class SequenceTaggerDecoderConfig(BaseDecoderConfig):
    @classmethod
    def module_name(cls):
        return "SequenceTaggerDecoder"

    type: str = schema_utils.ProtectedString(
        "tagger",
        description=DECODER_METADATA["SequenceTaggerDecoder"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["SequenceTaggerDecoder"]["input_size"],
    )

    vocab_size: int = common_fields.VocabSizeField()

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

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
