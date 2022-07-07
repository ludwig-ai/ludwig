from typing import ClassVar, Dict, List, Union, Callable
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

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="albert-base-v2",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30000,
        description="Vocabulary size of the ALBERT model. Defines the number of different tokens that can be "
                    "represented by the inputs_ids passed.",
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Dimensionality of vocabulary embeddings.",
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=4096,
        description="Dimensionality of the encoder layers and the pooler layer.",
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
    )

    num_hidden_groups: int = schema_utils.PositiveInteger(
        default=1,
        description="Number of groups for the hidden layers, parameters in the same group are shared.",
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=64,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=16384,
        description="The dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer "
                    "encoder.",
    )

    inner_group_num: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of inner repetition of attention and ffn.",
    )

    hidden_act: str = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu_new",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description="The dropout ratio for the attention probabilities.",
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
                    "something large (e.g., 512 or 1024 or 2048).",
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling AlbertModel or TFAlbertModel.",
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
    )

    classifier_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0.0,
        max=1.0,
        description="The dropout ratio for attached classifiers.",
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="",
    )

    pad_token_id: int = schema_utils.PositiveInteger(
        default=0,
        description="The ID of the token to use as padding.",
    )

    bos_token_id: int = schema_utils.PositiveInteger(
        default=2,
        description="The beginning of sequence token ID.",
    )

    eos_token_id: int = schema_utils.PositiveInteger(
        default=3,
        description="The end of sequence token ID.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class MT5EncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an MT5 encoder."""

    encoder_class: ClassVar[Encoder] = MT5Encoder

    type: str = "mt5"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="google/mt5-base",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=250112,
        description="Vocabulary size of the T5 model. Defines the number of different tokens that can be represented "
                    "by the inputs_ids passed when calling T5Model or TFT5Model.",
    )

    d_model: int = schema_utils.PositiveInteger(
        default=512,
        description="Size of the encoder layers and the pooler layer.",
    )

    d_kv: int = schema_utils.PositiveInteger(
        default=64,
        description="Size of the key, query, value projections per attention head. d_kv has to be equal to d_model // "
                    "num_heads.",
    )

    d_ff: int = schema_utils.PositiveInteger(
        default=1024,
        description="Size of the intermediate feed forward layer in each T5Block.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of hidden layers in the Transformer encoder.",
    )

    num_decoder_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of hidden layers in the Transformer decoder. Will use the same value as num_layers if not "
                    "set.",
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    relative_attention_num_buckets: int = schema_utils.PositiveInteger(
        default=32,
        description="The number of buckets to use for each attention layer.",
    )

    dropout_rate: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The ratio for all dropout layers.",
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-06,
        description="The epsilon used by the layer normalization layers.",
    )

    initializer_factor: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="A factor for initializing all weight matrices (should be kept to 1, used internally for "
                    "initialization testing)",
    )

    feed_forward_proj: str = schema_utils.StringOptions(
        ["relu", "gated-gelu"],
        default="gated-gelu",
        description="Type of feed forward layer to be used. ",
    )

    is_encoder_decoder: bool = schema_utils.Boolean(
        default=True,
        description="",
    )

    use_cache: bool = schema_utils.Boolean(
        default=True,
        description="",
    )

    tokenizer_class: str = schema_utils.String(
        default="T5Tokenizer",
        description="",
    )

    tie_word_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Whether the model's input and output word embeddings should be tied.",
    )

    pad_token_id: int = schema_utils.PositiveInteger(
        default=0,
        description="The ID of the token to use as padding.",
    )

    eos_token_id: int = schema_utils.PositiveInteger(
        default=1,
        description="The end of sequence token ID.",
    )

    decoder_start_token_id: int = schema_utils.PositiveInteger(
        default=0,
        description="If an encoder-decoder model starts decoding with a different token than _bos_, the id of that "
                    "token.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class XLMRoBERTaEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an XLMRoBERTa encoder."""

    encoder_class: ClassVar[Encoder] = XLMRoBERTaEncoder

    type: str = "xlmroberta"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlm-roberta-base",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Vocabulary size of the XLMRoBERTa model.",
    )

    pad_token_id: int = schema_utils.PositiveInteger(
        default=1,
        description="The ID of the token to use as padding.",
    )

    bos_token_id: int = schema_utils.PositiveInteger(
        default=0,
        description="The beginning of sequence token ID.",
    )

    eos_token_id: int = schema_utils.PositiveInteger(
        default=2,
        description="The end of sequence token ID.",
    )

    add_pooling_layer: bool = schema_utils.Boolean(
        default=True,
        description="Whether to add a pooling layer to the encoder.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class BERTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an BERT encoder."""

    encoder_class: ClassVar[Encoder] = BERTEncoder

    type: str = "bert"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="bert-base-uncased",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the BERT model. Defines the number of different tokens that can be "
                    "represented by the inputs_ids passed when calling BertModel or TFBertModel.",
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooler layer.",
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=3072,
        description="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
    )

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0.0,
        max=1.0,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0.0,
        max=1.0,
        description="The dropout ratio for the attention probabilities.",
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
                    "something large just in case (e.g., 512 or 1024 or 2048).",
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel.",
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
    )

    pad_token_id: int = schema_utils.PositiveInteger(
        default=0,
        description="The ID of the token to use as padding.",
    )

    gradient_checkpointing: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use gradient checkpointing.",
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="Type of position embedding.",
    )

    classifier_dropout: float = schema_utils.FloatRange(
        default=None,
        min=0.0,
        max=1.0,
        description="The dropout ratio for the classification head.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class XLMEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an XLM encoder."""

    encoder_class: ClassVar[Encoder] = XLMEncoder

    type: str = "xlm"

    max_sequence_length: int

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "xlm-mlm-en-2048"

    saved_weights_in_checkpoint: bool = False

    trainable: bool = False

    reduce_output: str = "cls_pooled"

    vocab_size: int = 30145

    emb_dim: int = 2048

    n_layers: int = 12

    n_heads: int = 16

    dropout: float = 0.1

    attention_dropout: float = 0.1

    gelu_activation: bool = True

    sinusoidal_embeddings: bool = False

    causal: bool = False

    asm: bool = False

    n_langs: int = 1

    use_lang_emb: bool = True

    max_position_embeddings: int = 512

    embed_init_std: float = 2048 ** -0.5

    layer_norm_eps: float = 1e-12

    init_std: float = 0.02

    bos_index: int = 0

    eos_index: int = 1

    pad_index: int = 2

    unk_index: int = 3

    mask_index: int = 5

    is_encoder: bool = True

    start_n_top: int = 5

    end_n_top: int = 5

    mask_token_id: int = 0

    lang_id: int = 0

    pad_token_id: int = 2

    bos_token_id: int = 0

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class GPTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an GPT encoder."""

    encoder_class: ClassVar[Encoder] = GPTEncoder

    type: str = "gpt"

    max_sequence_length: int

    reduce_output: str = "sum"

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "openai-gpt"

    saved_weights_in_checkpoint: bool = False

    trainable: bool = False

    vocab_size: int = 30522

    n_positions: int = 40478

    n_ctx: int = 512

    n_embd: int = 768

    n_layer: int = 12

    n_head: int = 12

    afn: str = "gelu"

    resid_pdrop: float = 0.1

    embd_pdrop: float = 0.1

    attn_pdrop: float = 0.1

    layer_norm_epsilon: float = 1e-5

    initializer_range: float = 0.02

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class GPT2EncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an GPT2 encoder."""

    encoder_class: ClassVar[Encoder] = GPT2Encoder

    type: str = "gpt2"

    max_sequence_length: int

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "gpt2"

    reduce_output: str = "sum"

    trainable: bool = False

    vocab_size: int = 50257

    n_positions: int = 1024

    n_ctx: int = 1024

    n_embd: int = 768

    n_layer: int = 12

    n_head: int = 12

    n_inner: int = None

    activation_function: str = "gelu"

    resid_pdrop: float = 0.1

    embd_pdrop: float = 0.1

    attn_pdrop: float = 0.1

    layer_norm_epsilon: float = 1e-5

    initializer_range: float = 0.02

    scale_attn_weights: bool = True

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class RoBERTaEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an RoBERTa encoder."""

    encoder_class: ClassVar[Encoder] = RoBERTaEncoder

    type: str = "roberta"

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "roberta-base"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "cls_pooled"

    trainable: bool = False

    vocab_size: int = None

    pad_token_id: int = 1

    bos_token_id: int = 0

    eos_token_id: int = 2

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class TransformerXLEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an TransformerXL encoder."""

    encoder_class: ClassVar[Encoder] = TransformerXLEncoder

    type: str = "transformer_xl"

    max_sequence_length: int

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "transfo-xl-wt103"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "sum"

    trainable: bool = False

    vocab_size: int = 267735

    cutoffs: List[int] = 20000, 40000, 200000

    d_model: int = 1024

    d_embed: int = 1024

    n_head: int = 16

    d_head: int = 64

    d_inner: int = 4096

    div_val: int = 4

    pre_lnorm: bool = False

    n_layer: int = 18

    mem_len: int = 1600

    clamp_len: int = 1000

    same_length: bool = True

    proj_share_all_but_first: bool = True

    attn_type: int = 0

    sample_softmax: int = -1

    adaptive: bool = True

    dropout: float = 0.1

    dropatt: float = 0.0

    untie_r: bool = True

    init: str = "normal"

    init_range: float = 0.01

    proj_init_std: float = 0.01

    init_std: float = 0.02

    layer_norm_epsilon: float = 1e-5

    eos_token_id: int = 0

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class XLNetEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an XLNet encoder."""

    encoder_class: ClassVar[Encoder] = XLNetEncoder

    type: str = "xlnet"

    max_sequence_length: int

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "xlnet-base-cased"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "sum"

    trainable: bool = False

    vocab_size: int = 32000

    d_model: int = 1024

    n_layer: int = 24

    n_head: int = 16

    d_inner: int = 4096

    ff_activation: str = "gelu"

    untie_r: bool = True

    attn_type: str = "bi"

    initializer_range: float = 0.02

    layer_norm_eps: float = 1e-12

    dropout: float = 0.1

    mem_len: int = 512

    reuse_len: int = None

    use_mems_eval: bool = True

    use_mems_train: bool = False

    bi_data: bool = False

    clamp_len: int = -1

    same_length: bool = False

    summary_type: str = "last"

    summary_use_proj: bool = True

    summary_activation: str = "tanh"

    summary_last_dropout: float = 0.1

    start_n_top: int = 5

    end_n_top: int = 5

    pad_token_id: int = 5

    bos_token_id: int = 1

    eos_token_id: int = 2

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class DistilBERTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an DistilBERT encoder."""

    encoder_class: ClassVar[Encoder] = DistilBERTEncoder

    type: str = "distilbert"

    max_sequence_length: int

    pretrained_model_name_or_path: str = "distilbert-base-uncased"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "sum"

    trainable: bool = True

    use_pretrained: bool = True

    vocab_size: int = 30522

    max_position_embeddings: int = 512

    sinusoidal_pos_embds: bool = False

    n_layers: int = 6

    n_heads: int = 12

    dim: int = 768

    hidden_dim: int = 3072

    dropout: float = 0.1

    attention_dropout: float = 0.1

    activation: Union[str, Callable] = "gelu"

    initializer_range: float = 0.02

    qa_dropout: float = 0.1

    seq_classif_dropout: float = 0.2

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class CTRLEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an CTRL encoder."""

    encoder_class: ClassVar[Encoder] = CTRLEncoder

    type: str = "ctrl"

    max_sequence_length: int

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "ctrl"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "sum"

    trainable: bool = True

    vocab_size: int = 246534

    n_positions: int = 256

    n_ctx: int = 256

    n_embd: int = 1280

    dff: int = 8192

    n_layer: int = 48

    n_head: int = 16

    resid_pdrop: float = 0.1

    embd_pdrop: float = 0.1

    attn_pdrop: float = 0.1

    layer_norm_epsilon: float = 1e-6

    initializer_range: float = 0.02

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class CamemBERTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an CamemBERT encoder."""

    encoder_class: ClassVar[Encoder] = CamemBERTEncoder

    type: str = "camembert"

    max_sequence_length: int

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "ctrl"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "cls-pooled"

    trainable: bool = False

    vocab_size: int = 30522

    hidden_size: int = 768

    num_hidden_layers: int = 12

    num_attention_heads: int = 12

    intermediate_size: int = 3072

    hidden_act: Union[str, Callable] = "gelu"

    hidden_dropout_prob: float = 0.1

    attention_probs_dropout_prob: float = 0.1

    max_position_embeddings: int = 512

    type_vocab_size: int = 2

    initializer_range: float = 0.02

    layer_norm_eps: float = 1e-12

    pad_token_id: int = 0

    gradient_checkpointing: bool = False

    position_embedding_type: str = "absolute"

    classifier_dropout: float = None

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class T5EncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an T5 encoder."""

    encoder_class: ClassVar[Encoder] = T5Encoder

    type: str = "t5"

    max_sequence_length: int

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "t5-small"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "sum"

    trainable: bool = False

    vocab_size: int = 32128

    d_model: int = 512

    d_kv: int = 64

    d_ff: int = 2048

    num_layers: int = 6

    num_decoder_layers: int = None

    num_heads: int = 8

    relative_attention_num_buckets: int = 32

    dropout_rate: float = 0.1

    layer_norm_eps: float = 1e-6

    initializer_factor: float = 1

    feed_forward_proj: str = "relu"

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class FlauBERTEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an FlauBERT encoder."""

    encoder_class: ClassVar[Encoder] = FlauBERTEncoder

    type: str = "flaubert"

    max_sequence_length: int

    use_pretrained: bool

    pretrained_model_name_or_path: str = "flaubert/flaubert_small_cased"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "sum"

    trainable: bool = False

    vocab_size: int = 30145

    pre_norm: bool = False

    layerdrop: float = 0.0

    emb_dim: int = 2048

    n_layer: int = 12

    n_head: int = 16

    dropout: float = 0.1

    attention_dropout: float = 0.1

    gelu_activation: bool = True

    sinusoidal_embeddings: bool = False

    causal: bool = False

    asm: bool = False

    n_langs: int = 1

    use_lang_emb: bool = True

    max_position_embeddings: int = 512

    embed_init_std: float = 2048 ** -0.5

    init_std: int = 50257

    layer_norm_eps: float = 1e-12

    bos_index: int = 0

    eos_index: int = 1

    pad_index: int = 2

    unk_index: int = 3

    mask_index: int = 5

    is_encoder: bool = True

    mask_token_id: int = 0

    lang_id: int = 1

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class ELECTRAEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an ELECTRA encoder."""

    encoder_class: ClassVar[Encoder] = ELECTRAEncoder

    type: str = "electra"

    max_sequence_length: int

    use_pretrained: bool = True

    pretrained_model_name_or_path: str = "google/electra-small-discriminator"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "sum"

    trainable: bool = False

    vocab_size: int = 30522

    embedding_size: int = 128

    hidden_size: int = 256

    num_hidden_layers: int = 12

    num_attention_heads: int = 4

    intermediate_size: int = 1024

    hidden_act: Union[str, Callable] = "gelu"

    hidden_dropout_prob: float = 0.1

    attention_probs_dropout_prob: float = 0.1

    max_position_embeddings: int = 512

    type_vocab_size: int = 2

    initializer_range: float = 0.02

    layer_norm_eps: float = 1e-12

    position_embedding_type: str = "absolute"

    classifier_dropout: float = None

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class LongformerEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an Longformer encoder."""

    encoder_class: ClassVar[Encoder] = LongformerEncoder

    type: str = "longformer"

    max_sequence_length: int

    use_pretrained: bool = True

    attention_window: Union[List[int], int] = 512

    sep_token_id: int = 2

    pretrained_model_name_or_path: str = "allenai/longformer-base-4096"

    saved_weights_in_checkpoint: bool = False

    reduce_output: str = "cls_pooled"

    trainable: bool = False

    num_tokens: int = None

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class AutoTransformerEncoderConfig(schema_utils.BaseMarshmallowConfig):
    """This dataclass configures the schema used for an AutoTransformer encoder."""

    encoder_class: ClassVar[Encoder] = AutoTransformerEncoder

    type: str = "auto_transformer"

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlm-roberta-base",
        description="Name or path of the pretrained model.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Vocabulary size of the XLMRoBERTa model.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )
