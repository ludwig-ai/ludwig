from typing import Optional, ClassVar
from ludwig.encoders.base import Encoder
from ludwig.encoders.text_encoders import (
    ALBERTEncoder,
    MT5Encoder,
    XLMRoBERTaEncoder,
    BERTEncoder,
    XLMEncoder,
    XLNetEncoder,
    GPTEncoder,
    GPT2Encoder,
    RoBERTaEncoder,
    TransformerXLEncoder,
    DistilBERTEncoder,
    CTRLEncoder,
    CamemBERTEncoder,
    T5Encoder,
    FlauBERTEncoder,
    ELECTRAEncoder,
    LongformerEncoder,
    AutoTransformerEncoder
)

from marshmallow_dataclass import dataclass
from ludwig.schema import utils as schema_utils


@dataclass
class ALBERTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an ALBERT encoder."""

    encoder_class: ClassVar[Encoder] = ALBERTEncoder

    type: str = "albert"

    max_sequence_length: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: Optional[str] = schema_utils.String(
        default="albert-base-v2",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    trainable: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    reduce_output: Optional[str] = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    vocab_size: Optional[int] = schema_utils.PositiveInteger(
        default=30000,
        description="Size of the vocabulary.",
    )

    embedding_size: Optional[int] = schema_utils.PositiveInteger(
        default=128,
        description="Size of the embedding.",
    )

    hidden_size: Optional[int] = schema_utils.PositiveInteger(
        default=4096,
        description="Size of the hidden layer.",
    )

    num_hidden_layers: Optional[int] = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers.",
    )

    num_hidden_groups: Optional[int] = schema_utils.PositiveInteger(
        default=1,
        description="Number of hidden groups.",
    )

    num_attention_heads: Optional[int] = schema_utils.PositiveInteger(
        default=64,
        description="Number of attention heads.",
    )

    intermediate_size: Optional[int] = schema_utils.PositiveInteger(
        default=16384,
        description="Size of the intermediate layer.",
    )

    inner_group_num: Optional[int] = schema_utils.PositiveInteger(
        default=1,
        description="",
    )

    hidden_act: Optional[str] = schema_utils.String(
        default="gelu_new",
        description="Activation function for the hidden layer.",
    )

    hidden_dropout_prob: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description="Dropout probability for the hidden layer.",
    )

    attention_probs_dropout_prob: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description="Dropout probability for the attention probabilities.",
    )

    max_position_embeddings: Optional[int] = schema_utils.PositiveInteger(
        default=512,
        description="Maximum position embeddings.",
    )

    type_vocab_size: Optional[int] = schema_utils.PositiveInteger(
        default=2,
        description="",
    )

    initializer_range: Optional[float] = schema_utils.NonNegativeFloat(
        default=0.02,
        description="",
    )

    layer_norm_eps: Optional[float] = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="",
    )

    classifier_dropout_prob: Optional[float] = schema_utils.FloatRange(
        default=0.1,
        min=0.0,
        max=1.0,
        description="",
    )

    position_embedding_type: Optional[str] = schema_utils.String(
        default="absolute",
        description="",
    )

    pad_token_id: Optional[int] = schema_utils.PositiveInteger(
        default=0,
        description="",
    )

    bos_token_id: Optional[int] = schema_utils.PositiveInteger(
        default=2,
        description="",
    )

    eos_token_id: Optional[int] = schema_utils.PositiveInteger(
        default=3,
        description="",
    )

    pretrained_kwargs: Optional[dict] = schema_utils.Dict(
        default=None,
        description="",
    )


@dataclass
class MT5EncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an MT5 encoder."""

    encoder_class: ClassVar[Encoder] = MT5Encoder

    type: str = "mt5"

    max_sequence_length: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: Optional[str] = schema_utils.String(
        default="google/mt5-base",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    trainable: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    reduce_output: Optional[str] = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    vocab_size: Optional[int] = schema_utils.PositiveInteger(
        default=250112,
        description="Size of the vocabulary.",
    )

    d_model: Optional[int] = schema_utils.PositiveInteger(
        default=512,
        description="",
    )

    d_kv: Optional[int] = schema_utils.PositiveInteger(
        default=64,
        description="",
    )

    d_ff: Optional[int] = schema_utils.PositiveInteger(
        default=1024,
        description="",
    )

    num_layers: Optional[int] = schema_utils.PositiveInteger(
        default=8,
        description="",
    )

    num_decoder_layers: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        description="",
    )

    num_heads: Optional[int] = schema_utils.PositiveInteger(
        default=6,
        description="",
    )

    relative_attention_num_buckets: int = 32,

    dropout_rate: float = 0.1,

    layer_norm_epsilon: float = 1e-06,

    initializer_factor: float = 1.0,

    feed_forward_proj: str = "gated-gelu",

    is_encoder_decoder: bool = True,

    use_cache: bool = True,

    tokenizer_class: str = "T5Tokenizer",

    tie_word_embeddings: bool = False,

    pad_token_id: int = 0,

    eos_token_id: int = 1,

    decoder_start_token_id: int = 0,

    pretrained_kwargs: Dict = None,


@dataclass
class XLMRoBERTaEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an XLMRoBERTa encoder."""

    encoder_class: ClassVar[Encoder] = XLMRoBERTaEncoder

    type: str = "xlmroberta"


@dataclass
class BERTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an BERT encoder."""

    encoder_class: ClassVar[Encoder] = BERTEncoder

    type: str = "bert"


@dataclass
class XLMEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an XLM encoder."""

    encoder_class: ClassVar[Encoder] = XLMEncoder

    type: str = "xlm"


@dataclass
class GPTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an GPT encoder."""

    encoder_class: ClassVar[Encoder] = GPTEncoder

    type: str = "gpt"


@dataclass
class GPT2EncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an GPT2 encoder."""

    encoder_class: ClassVar[Encoder] = GPT2Encoder

    type: str = "gpt2"


@dataclass
class RoBERTaEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an RoBERTa encoder."""

    encoder_class: ClassVar[Encoder] = RoBERTaEncoder

    type: str = "roberta"


@dataclass
class TransformerXLEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an TransformerXL encoder."""

    encoder_class: ClassVar[Encoder] = TransformerXLEncoder

    type: str = "transformer_xl"


@dataclass
class XLNetEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an XLNet encoder."""

    encoder_class: ClassVar[Encoder] = XLNetEncoder

    type: str = "xlnet"


@dataclass
class DistilBERTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an DistilBERT encoder."""

    encoder_class: ClassVar[Encoder] = DistilBERTEncoder

    type: str = "distilbert"


@dataclass
class CTRLEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an CTRL encoder."""

    encoder_class: ClassVar[Encoder] = CTRLEncoder

    type: str = "ctrl"


@dataclass
class CamemBERTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an CamemBERT encoder."""

    encoder_class: ClassVar[Encoder] = CamemBERTEncoder

    type: str = "camembert"


@dataclass
class T5EncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an T5 encoder."""

    encoder_class: ClassVar[Encoder] = T5Encoder

    type: str = "t5"


@dataclass
class FlauBERTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an FlauBERT encoder."""

    encoder_class: ClassVar[Encoder] = FlauBERTEncoder

    type: str = "flaubert"


@dataclass
class ELECTRAEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an ELECTRA encoder."""

    encoder_class: ClassVar[Encoder] = ELECTRAEncoder

    type: str = "electra"


@dataclass
class LongformerEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an Longformer encoder."""

    encoder_class: ClassVar[Encoder] = LongformerEncoder

    type: str = "longformer"


@dataclass
class AutoTransformerEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an AutoTransformer encoder."""

    encoder_class: ClassVar[Encoder] = AutoTransformerEncoder

    type: str = "auto_transformer"
