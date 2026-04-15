from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MODEL_ECD, SEQUENCE, TEXT
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.metadata import DECODER_METADATA


@DeveloperAPI
@register_decoder_config("generator", [SEQUENCE, TEXT], model_types=[MODEL_ECD])
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

    # Scheduled sampling (Bengio et al., NeurIPS 2015)
    teacher_forcing_decay: str = schema_utils.StringOptions(
        ["none", "linear", "exponential"],
        default="none",
        description=(
            "Decay schedule for teacher forcing probability during training. "
            "none always uses full teacher forcing; linear linearly decays the probability to zero; "
            "exponential applies exponential decay. "
            "Implements scheduled sampling (Bengio et al., NeurIPS 2015)."
        ),
    )

    teacher_forcing_decay_rate: float = schema_utils.FloatRange(
        default=0.01,
        min=0.0,
        max=1.0,
        description=(
            "Rate of decay for the teacher forcing probability per decoding step when "
            "teacher_forcing_decay is linear or exponential."
        ),
    )

    # Beam search
    beam_width: int = schema_utils.PositiveInteger(
        default=1,
        description=(
            "Width of the beam for beam search decoding. 1 = greedy decoding (default). "
            "Values > 1 enable beam search at inference time, keeping the top beam_width "
            "candidate sequences at each step."
        ),
    )

    beam_length_penalty: float = schema_utils.FloatRange(
        default=1.0,
        min=0.0,
        description=(
            "Length penalty exponent applied to beam search scores. "
            "Score = log_prob / (length ^ beam_length_penalty). "
            "Values > 1 penalise longer sequences; values < 1 favour them. "
            "Only used when beam_width > 1."
        ),
    )


@DeveloperAPI
@register_decoder_config("transformer_generator", [SEQUENCE, TEXT], model_types=[MODEL_ECD])
class TransformerDecoderConfig(BaseDecoderConfig):
    """Configuration for the Transformer-based sequence/text decoder.

    References:
        Vaswani et al., Attention Is All You Need, NeurIPS 2017.
        https://arxiv.org/abs/1706.03762
    """

    @staticmethod
    def module_name():
        return "SequenceTransformerDecoder"

    type: str = schema_utils.ProtectedString(
        "transformer_generator",
        description=(
            "Transformer-based autoregressive sequence decoder. "
            "Uses teacher forcing during training and autoregressive generation at inference. "
            "Based on Vaswani et al., Attention Is All You Need, NeurIPS 2017."
        ),
    )

    vocab_size: int = common_fields.VocabSizeField()

    max_sequence_length: int = common_fields.MaxSequenceLengthField()

    input_size: int = schema_utils.PositiveInteger(
        default=256,
        description=(
            "Size of the encoder output (d_model). The encoder output is projected to this size "
            "before being used as cross-attention memory in the transformer decoder."
        ),
    )

    d_model: int = schema_utils.PositiveInteger(
        default=256,
        description=(
            "Dimensionality of the transformer decoder layers (embedding size and hidden size). "
            "Must match input_size or a projection will be applied."
        ),
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=2,
        description="Number of transformer decoder layers.",
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description=(
            "Number of attention heads in each multi-head attention sub-layer. "
            "d_model must be divisible by num_heads."
        ),
    )

    ffn_size: int = schema_utils.PositiveInteger(
        default=1024,
        description="Size of the feed-forward network (dim_feedforward) inside each transformer decoder layer.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0.0,
        max=1.0,
        description="Dropout probability applied within transformer decoder layers.",
    )

    reduce_input: str = schema_utils.StringOptions(
        ["sum", "mean", "avg", "max", "concat", "last"],
        default="sum",
        description=(
            "How to reduce a 3-D encoder output (batch x seq x hidden) to a 2-D context vector "
            "(batch x hidden). Ignored when the encoder output is already 2-D."
        ),
    )

    # Beam search
    beam_width: int = schema_utils.PositiveInteger(
        default=1,
        description=(
            "Width of the beam for beam search decoding at inference time. "
            "1 = greedy decoding (default). Values > 1 keep the top beam_width candidates at each step."
        ),
    )

    beam_length_penalty: float = schema_utils.FloatRange(
        default=1.0,
        min=0.0,
        description=(
            "Length penalty exponent for beam search. "
            "Score = log_prob / (length ^ beam_length_penalty). "
            "Only active when beam_width > 1."
        ),
    )


@DeveloperAPI
@register_decoder_config("tagger", [SEQUENCE, TEXT], model_types=[MODEL_ECD])
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
