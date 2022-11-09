from typing import Callable, List, Union

from marshmallow_dataclass import dataclass

from ludwig.constants import TEXT
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata.encoder_metadata import ENCODER_METADATA
from ludwig.schema.metadata.parameter_metadata import ParameterMetadata


@register_encoder_config("albert", TEXT)
@dataclass(repr=False)
class ALBERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an ALBERT encoder."""

    type: str = schema_utils.StringOptions(
        ["albert"],
        default="albert",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="albert-base-v2",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["trainable"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["reduce_output"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30000,
        description="Vocabulary size of the ALBERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["vocab_size"],
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Dimensionality of vocabulary embeddings.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["embedding_size"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=4096,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["hidden_size"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["num_hidden_layers"],
    )

    num_hidden_groups: int = schema_utils.PositiveInteger(
        default=1,
        description="Number of groups for the hidden layers, parameters in the same group are shared.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["num_hidden_groups"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=64,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["num_attention_heads"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=16384,
        description="The dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer "
        "encoder.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["intermediate_size"],
    )

    inner_group_num: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of inner repetition of attention and ffn.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["inner_group_num"],
    )

    hidden_act: str = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu_new",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["hidden_dropout_prob"],
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["attention_probs_dropout_prob"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling AlbertModel or TFAlbertModel.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["type_vocab_size"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["layer_norm_eps"],
    )

    classifier_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for attached classifiers.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["classifier_dropout_prob"],
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["position_embedding_type"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=2,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["bos_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=3,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["eos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["ALBERTEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("mt5", TEXT)
@dataclass(repr=False)
class MT5Config(BaseEncoderConfig):
    """This dataclass configures the schema used for an MT5 encoder."""

    type: str = schema_utils.StringOptions(
        ["mt5"],
        default="mt5",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="google/mt5-base",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["trainable"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["reduce_output"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=250112,
        description="Vocabulary size of the T5 model. Defines the number of different tokens that can be represented "
        "by the inputs_ids passed when calling T5Model or TFT5Model.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["vocab_size"],
    )

    d_model: int = schema_utils.PositiveInteger(
        default=512,
        description="Size of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["d_model"],
    )

    d_kv: int = schema_utils.PositiveInteger(
        default=64,
        description="Size of the key, query, value projections per attention head. d_kv has to be equal to d_model // "
        "num_heads.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["d_kv"],
    )

    d_ff: int = schema_utils.PositiveInteger(
        default=1024,
        description="Size of the intermediate feed forward layer in each T5Block.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["d_ff"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["num_layers"],
    )

    num_decoder_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of hidden layers in the Transformer decoder. Will use the same value as num_layers if not "
        "set.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["num_decoder_layers"],
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["num_heads"],
    )

    relative_attention_num_buckets: int = schema_utils.PositiveInteger(
        default=32,
        description="The number of buckets to use for each attention layer.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["relative_attention_num_buckets"],
    )

    dropout_rate: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The ratio for all dropout layers.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["dropout_rate"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-06,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["layer_norm_epsilon"],
    )

    initializer_factor: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="A factor for initializing all weight matrices (should be kept to 1, used internally for "
        "initialization testing)",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["initializer_factor"],
    )

    feed_forward_proj: str = schema_utils.StringOptions(
        ["relu", "gated-gelu"],
        default="gated-gelu",
        description="Type of feed forward layer to be used. ",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["feed_forward_proj"],
    )

    is_encoder_decoder: bool = schema_utils.Boolean(
        default=True,
        description="",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["is_encoder_decoder"],
    )

    use_cache: bool = schema_utils.Boolean(
        default=True,
        description="",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["use_cache"],
    )

    tokenizer_class: str = schema_utils.String(
        default="T5Tokenizer",
        description="",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["tokenizer_class"],
    )

    tie_word_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Whether the model's input and output word embeddings should be tied.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["tie_word_embeddings"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["pad_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=1,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["eos_token_id"],
    )

    decoder_start_token_id: int = schema_utils.Integer(
        default=0,
        description="If an encoder-decoder model starts decoding with a different token than _bos_, the id of that "
        "token.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["decoder_start_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["MT5Encoder"]["pretrained_kwargs"],
    )


@register_encoder_config("xlmroberta", TEXT)
@dataclass(repr=False)
class XLMRoBERTaConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an XLMRoBERTa encoder."""

    type: str = schema_utils.StringOptions(
        ["xlmroberta"],
        default="xlmroberta",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlm-roberta-base",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Vocabulary size of the XLMRoBERTa model.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["vocab_size"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=1,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=0,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["bos_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=2,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["eos_token_id"],
    )

    add_pooling_layer: bool = schema_utils.Boolean(
        default=True,
        description="Whether to add a pooling layer to the encoder.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["add_pooling_layer"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTaEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("bert", TEXT)
@dataclass(repr=False)
class BERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an BERT encoder."""

    type: str = schema_utils.StringOptions(
        ["bert"],
        default="bert",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="bert-base-uncased",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["trainable"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["reduce_output"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the BERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling BertModel or TFBertModel.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["vocab_size"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["hidden_size"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["num_hidden_layers"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["num_attention_heads"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=3072,
        description="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["intermediate_size"],
    )

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(  # TODO: add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["hidden_dropout_prob"],
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["attention_probs_dropout_prob"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["type_vocab_size"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["layer_norm_eps"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["pad_token_id"],
    )

    gradient_checkpointing: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use gradient checkpointing.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["gradient_checkpointing"],
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="Type of position embedding.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["position_embedding_type"],
    )

    classifier_dropout: float = schema_utils.FloatRange(
        default=None,
        min=0,
        max=1,
        description="The dropout ratio for the classification head.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["classifier_dropout"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["BERTEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("xlm", TEXT)
@dataclass(repr=False)
class XLMConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an XLM encoder."""

    type: str = schema_utils.StringOptions(
        ["xlm"],
        default="xlm",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlm-mlm-en-2048",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["trainable"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["reduce_output"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30145,
        description="Vocabulary size of the BERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling XLMModel or TFXLMModel.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["vocab_size"],
    )

    emb_dim: int = schema_utils.PositiveInteger(
        default=2048,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["emb_dim"],
    )

    n_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["n_layers"],
    )

    n_heads: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["n_heads"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["dropout"],
    )

    attention_dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for the attention mechanism.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["attention_dropout"],
    )

    gelu_activation: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use gelu for the activations instead of relu.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["gelu_activation"],
    )

    sinusoidal_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["sinusoidal_embeddings"],
    )

    causal: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not the model should behave in a causal manner. Causal models use a triangular "
        "attention mask in order to only attend to the left-side context instead if a bidirectional "
        "context.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["causal"],
    )

    asm: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the "
        "prediction layer.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["asm"],
    )

    n_langs: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of languages the model handles. Set to 1 for monolingual models.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["n_langs"],
    )

    use_lang_emb: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use language embeddings. Some models use additional language embeddings, "
        "see the multilingual models page for information on how to use them.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["use_lang_emb"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["max_position_embeddings"],
    )

    embed_init_std: float = schema_utils.NonNegativeFloat(
        default=2048**-0.5,
        description="The standard deviation of the truncated_normal_initializer for initializing the embedding "
        "matrices.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["embed_init_std"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["layer_norm_eps"],
    )

    init_std: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices "
        "except the embedding matrices.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["init_std"],
    )

    bos_index: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The index of the beginning of sentence token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["bos_index"],
    )

    eos_index: int = schema_utils.NonNegativeInteger(
        default=1,
        description="The index of the end of sentence token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["eos_index"],
    )

    pad_index: int = schema_utils.NonNegativeInteger(
        default=2,
        description="The index of the padding token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["pad_index"],
    )

    unk_index: int = schema_utils.NonNegativeInteger(
        default=3,
        description="The index of the unknown token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["unk_index"],
    )

    mask_index: int = schema_utils.NonNegativeInteger(
        default=5,
        description="The index of the masking token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["mask_index"],
    )

    is_encoder: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not the initialized model should be a transformer encoder or decoder as seen in "
        "Vaswani et al.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["is_encoder"],
    )

    start_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description="Used in the SQuAD evaluation script.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["start_n_top"],
    )

    end_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description="Used in the SQuAD evaluation script.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["end_n_top"],
    )

    mask_token_id: int = schema_utils.Integer(
        default=0,
        description="Model agnostic parameter to identify masked tokens when generating text in an MLM context.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["mask_token_id"],
    )

    lang_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the language used by the model. This parameter is used when generating text in a given "
        "language.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["lang_id"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=2,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=0,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["bos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLMEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("gpt", TEXT)
@dataclass(repr=False)
class GPTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an GPT encoder."""

    type: str = schema_utils.StringOptions(
        ["gpt"],
        default="gpt",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["max_sequence_length"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["reduce_output"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="openai-gpt",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the GPT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling OpenAIGPTModel or TFOpenAIGPTModel.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["vocab_size"],
    )

    n_positions: int = schema_utils.PositiveInteger(
        default=40478,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["n_positions"],
    )

    n_ctx: int = schema_utils.PositiveInteger(
        default=512,
        description="Dimensionality of the causal mask (usually same as n_positions)",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["n_ctx"],
    )

    n_embd: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the embeddings and hidden states.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["n_embd"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["n_layer"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["n_head"],
    )

    afn: str = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu_new",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["afn"],
    )

    resid_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["resid_pdrop"],
    )

    embd_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout ratio for the embeddings.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["embd_pdrop"],
    )

    attn_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout ratio for the attention.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["attn_pdrop"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-5,
        description="The epsilon to use in the layer normalization layers",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["layer_norm_epsilon"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["initializer_range"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["GPTEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("gpt2", TEXT)
@dataclass(repr=False)
class GPT2Config(BaseEncoderConfig):
    """This dataclass configures the schema used for an GPT2 encoder."""

    type: str = schema_utils.StringOptions(
        ["gpt2"],
        default="gpt2",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="gpt2",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["pretrained_model_name_or_path"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=50257,
        description="Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling GPT2Model or TFGPT2Model.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["vocab_size"],
    )

    n_positions: int = schema_utils.PositiveInteger(
        default=1024,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["n_positions"],
    )

    n_ctx: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the causal mask (usually same as n_positions)",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["n_ctx"],
    )

    n_embd: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the embeddings and hidden states.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["n_embd"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["n_layer"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["n_head"],
    )

    n_inner: int = schema_utils.PositiveInteger(
        default=None,
        description="Dimensionality of the inner feed-forward layers. None will set it to 4 times n_embd",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["n_inner"],
    )

    activation_function: str = schema_utils.StringOptions(
        ["relu", "silu", "gelu", "tanh", "gelu_new"],
        default="gelu",
        description="Activation function, to be selected in the list ['relu', 'silu', 'gelu', 'tanh', 'gelu_new'].",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["activation_function"],
    )

    resid_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["resid_pdrop"],
    )

    embd_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the embeddings.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["embd_pdrop"],
    )

    attn_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["attn_pdrop"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-5,
        description="The epsilon to use in the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["layer_norm_epsilon"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["initializer_range"],
    )

    scale_attn_weights: bool = schema_utils.Boolean(
        default=True,
        description="Scale attention weights by dividing by sqrt(hidden_size).",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["scale_attn_weights"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["GPT2Encoder"]["pretrained_kwargs"],
    )


@register_encoder_config("roberta", TEXT)
@dataclass(repr=False)
class RoBERTaConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an RoBERTa encoder."""

    type: str = schema_utils.StringOptions(
        ["roberta"],
        default="roberta",
        allow_none=False,
        description="Type of encoder.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="roberta-base",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Vocabulary size of the RoBERTa model.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["vocab_size"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=1,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=0,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["bos_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=2,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["eos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["RoBERTaEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("transformer_xl", TEXT)
@dataclass(repr=False)
class TransformerXLConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an TransformerXL encoder."""

    type: str = schema_utils.StringOptions(
        ["transformer_xl"],
        default="transformer_xl",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="transfo-xl-wt103",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=267735,
        description="Vocabulary size of the TransfoXL model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling TransfoXLModel or TFTransfoXLModel.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["vocab_size"],
    )

    cutoffs: List[int] = schema_utils.List(
        int,
        default=[20000, 40000, 200000],
        description="Cutoffs for the adaptive softmax.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["cutoffs"],
    )

    d_model: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the model’s hidden states.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["d_model"],
    )

    d_embed: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the embeddings",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["d_embed"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["n_head"],
    )

    d_head: int = schema_utils.PositiveInteger(
        default=64,
        description="Dimensionality of the model’s heads.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["d_head"],
    )

    d_inner: int = schema_utils.PositiveInteger(
        default=4096,
        description=" Inner dimension in FF",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["d_inner"],
    )

    div_val: int = schema_utils.PositiveInteger(
        default=4,
        description="Divident value for adapative input and softmax.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["div_val"],
    )

    pre_lnorm: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to apply LayerNorm to the input instead of the output in the blocks.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["pre_lnorm"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=18,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["n_layer"],
    )

    mem_len: int = schema_utils.PositiveInteger(
        default=1600,
        description="Length of the retained previous heads.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["mem_len"],
    )

    clamp_len: int = schema_utils.PositiveInteger(
        default=1000,
        description="Use the same pos embeddings after clamp_len.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["clamp_len"],
    )

    same_length: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use the same attn length for all tokens",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["same_length"],
    )

    proj_share_all_but_first: bool = schema_utils.Boolean(
        default=True,
        description="True to share all but first projs, False not to share.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["proj_share_all_but_first"],
    )

    attn_type: int = schema_utils.IntegerRange(
        default=0,
        min=0,
        max=3,
        description="Attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["attn_type"],
    )

    sample_softmax: int = schema_utils.Integer(
        default=-1,
        description="Number of samples in the sampled softmax.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["sample_softmax"],
    )

    adaptive: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use adaptive softmax.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["adaptive"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["dropout"],
    )

    dropatt: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["dropatt"],
    )

    untie_r: bool = schema_utils.Boolean(
        default=True,
        description="Whether ot not to untie relative position biases.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["untie_r"],
    )

    init: str = schema_utils.String(
        default="normal",
        description="Parameter initializer to use.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["init"],
    )

    init_range: float = schema_utils.NonNegativeFloat(
        default=0.01,
        description="Parameters initialized by U(-init_range, init_range).",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["init_range"],
    )

    proj_init_std: float = schema_utils.NonNegativeFloat(
        default=0.01,
        description="Parameters initialized by N(0, init_std)",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["proj_init_std"],
    )

    init_std: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="Parameters initialized by N(0, init_std)",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["init_std"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-5,
        description="The epsilon to use in the layer normalization layers",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["layer_norm_epsilon"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=0,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["eos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["TransformerXLEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("xlnet", TEXT)
@dataclass(repr=False)
class XLNetConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an XLNet encoder."""

    type: str = schema_utils.StringOptions(
        ["xlnet"],
        default="xlnet",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlnet-base-cased",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=32000,
        description="Vocabulary size of the XLNet model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling XLNetModel or TFXLNetModel.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["vocab_size"],
    )

    d_model: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["d_model"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=24,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["n_layer"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["n_head"],
    )

    d_inner: int = schema_utils.PositiveInteger(
        default=4096,
        description="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["d_inner"],
    )

    ff_activation: str = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler. If string, "
        "'gelu', 'relu', 'silu' and 'gelu_new' are supported.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["ff_activation"],
    )

    untie_r: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to untie relative position biases",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["untie_r"],
    )

    attn_type: str = schema_utils.StringOptions(
        ["bi", "uni"],
        default="bi",
        description="The attention type used by the model. Set 'bi' for XLNet, 'uni' for Transformer-XL.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["attn_type"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["layer_norm_eps"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["dropout"],
    )

    mem_len: int = schema_utils.PositiveInteger(
        default=512,
        description="The number of tokens to cache. The key/value pairs that have already been pre-computed in a "
        "previous forward pass won’t be re-computed. ",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["mem_len"],
    )

    reuse_len: int = schema_utils.PositiveInteger(
        default=None,
        description="The number of tokens in the current batch to be cached and reused in the future.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["reuse_len"],
    )

    use_mems_eval: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["use_mems_eval"],
    )

    use_mems_train: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not the model should make use of the recurrent memory mechanism in train mode.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["use_mems_train"],
    )

    bi_data: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use bidirectional input pipeline. Usually set to True during pretraining and "
        "False during finetuning.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["bi_data"],
    )

    clamp_len: int = schema_utils.Integer(
        default=-1,
        description="Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no "
        "clamping.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["clamp_len"],
    )

    same_length: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use the same attention length for each token.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["same_length"],
    )

    summary_type: str = schema_utils.StringOptions(
        ["last", "first", "mean", "cls_index", "attn"],
        default="last",
        description="Argument used when doing sequence summary. Used in the sequence classification and multiple "
        "choice models.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["summary_type"],
    )

    summary_use_proj: bool = schema_utils.Boolean(
        default=True,
        description="",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["summary_use_proj"],
    )

    summary_activation: str = schema_utils.String(
        default="tanh",
        description="Argument used when doing sequence summary. Used in the sequence classification and multiple "
        "choice models.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["summary_activation"],
    )

    summary_last_dropout: float = schema_utils.FloatRange(
        default=0.1,
        description="Used in the sequence classification and multiple choice models.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["summary_last_dropout"],
    )

    start_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description="Used in the SQuAD evaluation script.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["start_n_top"],
    )

    end_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description=" Used in the SQuAD evaluation script.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["end_n_top"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=5,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=1,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["bos_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=2,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["eos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLNetEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("distilbert", TEXT)
@dataclass(repr=False)
class DistilBERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an DistilBERT encoder."""

    type: str = schema_utils.StringOptions(
        ["distilbert"],
        default="distilbert",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["max_sequence_length"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="distilbert-base-uncased",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["trainable"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["use_pretrained"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling DistilBertModel or TFDistilBertModel.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["vocab_size"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["max_position_embeddings"],
    )

    sinusoidal_pos_embds: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use sinusoidal positional embeddings.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["sinusoidal_pos_embds"],
    )

    n_layers: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["n_layers"],
    )

    n_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["n_heads"],
    )

    dim: int = schema_utils.PositiveInteger(
        default=768,
        description=" Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["dim"],
    )

    hidden_dim: int = schema_utils.PositiveInteger(
        default=3072,
        description="The size of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["hidden_dim"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["dropout"],
    )

    attention_dropout: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["attention_dropout"],
    )

    activation: Union[str, Callable] = schema_utils.StringOptions(  # TODO: Add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler. If string, "
        "'gelu', 'relu', 'silu' and 'gelu_new' are supported.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["activation"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["initializer_range"],
    )

    qa_dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probabilities used in the question answering model DistilBertForQuestionAnswering.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["qa_dropout"],
    )

    seq_classif_dropout: float = schema_utils.FloatRange(
        default=0.2,
        min=0,
        max=1,
        description="The dropout probabilities used in the sequence classification and the multiple choice model "
        "DistilBertForSequenceClassification.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["seq_classif_dropout"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["DistilBERTEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("ctrl", TEXT)
@dataclass(repr=False)
class CTRLConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an CTRL encoder."""

    type: str = schema_utils.StringOptions(
        ["ctrl"],
        default="ctrl",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="ctrl",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=246534,
        description="Vocabulary size of the CTRL model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling CTRLModel or TFCTRLModel.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["vocab_size"],
    )

    n_positions: int = schema_utils.PositiveInteger(
        default=256,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["n_positions"],
    )

    n_ctx: int = schema_utils.PositiveInteger(
        default=256,
        description="Dimensionality of the causal mask (usually same as n_positions)",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["n_ctx"],
    )

    n_embd: int = schema_utils.PositiveInteger(
        default=1280,
        description="Dimensionality of the embeddings and hidden states.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["n_embd"],
    )

    dff: int = schema_utils.PositiveInteger(
        default=8192,
        description="Dimensionality of the inner dimension of the feed forward networks (FFN).",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["dff"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=48,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["n_layer"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["n_head"],
    )

    resid_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description=" The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["resid_pdrop"],
    )

    embd_pdrop: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the embeddings.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["embd_pdrop"],
    )

    attn_pdrop: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the attention.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["attn_pdrop"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-6,
        description="The epsilon to use in the layer normalization layers",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["layer_norm_epsilon"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["initializer_range"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["CTRLEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("camembert", TEXT)
@dataclass(repr=False)
class CamemBERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an CamemBERT encoder."""

    type: str = schema_utils.StringOptions(
        ["camembert"],
        default="camembert",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["use_pretrained"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["saved_weights_in_checkpoint"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="jplu/camembert-base",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["pretrained_model_name_or_path"],
    )

    reduce_output: str = schema_utils.String(
        default="cls-pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the CamemBERT model.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["vocab_size"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["hidden_size"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["num_hidden_layers"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["num_attention_heads"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=3072,
        description="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["intermediate_size"],
    )

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(  # TODO: add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["hidden_dropout_prob"],
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["attention_probs_dropout_prob"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["type_vocab_size"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["layer_norm_eps"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["pad_token_id"],
    )

    gradient_checkpointing: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use gradient checkpointing.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["gradient_checkpointing"],
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="Type of position embedding.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["position_embedding_type"],
    )

    classifier_dropout: float = schema_utils.FloatRange(
        default=None,
        min=0,
        max=1,
        description="The dropout ratio for the classification head.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["classifier_dropout"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["CamemBERTEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("t5", TEXT)
@dataclass(repr=False)
class T5Config(BaseEncoderConfig):
    """This dataclass configures the schema used for an T5 encoder."""

    type: str = schema_utils.StringOptions(
        ["t5"],
        default="t5",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="t5-small",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=32128,
        description="Vocabulary size of the T5 model. Defines the number of different tokens that can be represented "
        "by the inputs_ids passed when calling T5Model or TFT5Model.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["vocab_size"],
    )

    d_model: int = schema_utils.PositiveInteger(
        default=512,
        description="Size of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["d_model"],
    )

    d_kv: int = schema_utils.PositiveInteger(
        default=64,
        description="Size of the key, query, value projections per attention head. d_kv has to be equal to d_model // "
        "num_heads.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["d_kv"],
    )

    d_ff: int = schema_utils.PositiveInteger(
        default=2048,
        description="Size of the intermediate feed forward layer in each T5Block.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["d_ff"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["num_layers"],
    )

    num_decoder_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of hidden layers in the Transformer decoder. Will use the same value as num_layers if not "
        "set.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["num_decoder_layers"],
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["num_heads"],
    )

    relative_attention_num_buckets: int = schema_utils.PositiveInteger(
        default=32,
        description="The number of buckets to use for each attention layer.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["relative_attention_num_buckets"],
    )

    dropout_rate: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The ratio for all dropout layers.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["dropout_rate"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-6,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["layer_norm_eps"],
    )

    initializer_factor: float = schema_utils.NonNegativeFloat(
        default=1,
        description="A factor for initializing all weight matrices (should be kept to 1, used internally for "
        "initialization testing).",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["initializer_factor"],
    )

    feed_forward_proj: str = schema_utils.StringOptions(
        ["relu", "gated-gelu"],
        default="relu",
        description="Type of feed forward layer to be used. Should be one of 'relu' or 'gated-gelu'. T5v1.1 uses the "
        "'gated-gelu' feed forward projection. Original T5 uses 'relu'.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["feed_forward_proj"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["T5Encoder"]["pretrained_kwargs"],
    )


@register_encoder_config("flaubert", TEXT)
@dataclass(repr=False)
class FlauBERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an FlauBERT encoder."""

    type: str = schema_utils.StringOptions(
        ["flaubert"],
        default="flaubert",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="t5-small",
        description="flaubert/flaubert_small_cased",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30145,
        description="Vocabulary size of the FlauBERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling FlaubertModel or TFFlaubertModel.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["vocab_size"],
    )

    pre_norm: bool = schema_utils.Boolean(
        default=False,
        description="Whether to apply the layer normalization before or after the feed forward layer following the "
        "attention in each layer (Vaswani et al., Tensor2Tensor for Neural Machine Translation. 2018)",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["pre_norm"],
    )

    layerdrop: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Probability to drop layers during training (Fan et al., Reducing Transformer Depth on Demand "
        "with Structured Dropout. ICLR 2020)",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["layerdrop"],
    )

    emb_dim: int = schema_utils.PositiveInteger(
        default=2048,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["emb_dim"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["n_layer"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["n_head"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["dropout"],
    )

    attention_dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for the attention mechanism",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["attention_dropout"],
    )

    gelu_activation: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use a gelu activation instead of relu.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["gelu_activation"],
    )

    sinusoidal_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["sinusoidal_embeddings"],
    )

    causal: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not the model should behave in a causal manner. Causal models use a triangular "
        "attention mask in order to only attend to the left-side context instead if a bidirectional "
        "context.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["causal"],
    )

    asm: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the "
        "prediction layer.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["asm"],
    )

    n_langs: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of languages the model handles. Set to 1 for monolingual models.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["n_langs"],
    )

    use_lang_emb: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use language embeddings. Some models use additional language embeddings, "
        "see the multilingual models page for information on how to use them.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["use_lang_emb"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["max_position_embeddings"],
    )

    embed_init_std: float = schema_utils.NonNegativeFloat(
        default=2048**-0.5,
        description="The standard deviation of the truncated_normal_initializer for initializing the embedding "
        "matrices.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["embed_init_std"],
    )

    init_std: int = schema_utils.PositiveInteger(
        default=50257,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices "
        "except the embedding matrices.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["init_std"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["layer_norm_eps"],
    )

    bos_index: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The index of the beginning of sentence token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["bos_index"],
    )

    eos_index: int = schema_utils.NonNegativeInteger(
        default=1,
        description="The index of the end of sentence token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["eos_index"],
    )

    pad_index: int = schema_utils.NonNegativeInteger(
        default=2,
        description="The index of the padding token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["pad_index"],
    )

    unk_index: int = schema_utils.NonNegativeInteger(
        default=3,
        description="The index of the unknown token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["unk_index"],
    )

    mask_index: int = schema_utils.NonNegativeInteger(
        default=5,
        description="The index of the masking token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["mask_index"],
    )

    is_encoder: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not the initialized model should be a transformer encoder or decoder as seen in "
        "Vaswani et al.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["is_encoder"],
    )

    mask_token_id: int = schema_utils.Integer(
        default=0,
        description="Model agnostic parameter to identify masked tokens when generating text in an MLM context.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["mask_token_id"],
    )

    lang_id: int = schema_utils.Integer(
        default=1,
        description="The ID of the language used by the model. This parameter is used when generating text in a given "
        "language.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["lang_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["FlauBERTEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("electra", TEXT)
@dataclass(repr=False)
class ELECTRAConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an ELECTRA encoder."""

    type: str = schema_utils.StringOptions(
        ["electra"],
        default="electra",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="google/electra-small-discriminator",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the ELECTRA model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling ElectraModel or TFElectraModel.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["vocab_size"],
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["embedding_size"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["hidden_size"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["num_hidden_layers"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=4,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["num_attention_heads"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["intermediate_size"],
    )

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(  # TODO: add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["hidden_dropout_prob"],
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["attention_probs_dropout_prob"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling ElectraModel or TFElectraModel.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["type_vocab_size"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["layer_norm_eps"],
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="Type of position embedding.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["position_embedding_type"],
    )

    classifier_dropout: float = schema_utils.FloatRange(
        default=None,
        min=0,
        max=1,
        description="The dropout ratio for the classification head.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["classifier_dropout"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["ELECTRAEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("longformer", TEXT)
@dataclass(repr=False)
class LongformerConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an Longformer encoder."""

    type: str = schema_utils.StringOptions(
        ["longformer"],
        default="longformer",
        allow_none=False,
        description="Type of encoder.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["use_pretrained"],
    )

    attention_window: Union[List[int], int] = schema_utils.OneOfOptionsField(
        default=512,
        description="Size of an attention window around each token. If an int, use the same size for all layers. To "
        "specify a different window size for each layer, use a List[int] where len(attention_window) == "
        "num_hidden_layers.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=False, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["attention_window"],
    )

    sep_token_id: int = schema_utils.Integer(
        default=2,
        description="ID of the separator token, which is used when building a sequence from multiple sequences",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["sep_token_id"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="allenai/longformer-base-4096",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Vocabulary size of the Longformer model.",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["vocab_size"],
    )

    num_tokens: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of tokens",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["num_tokens"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["LongformerEncoder"]["pretrained_kwargs"],
    )


@register_encoder_config("auto_transformer", TEXT)
@dataclass(repr=False)
class AutoTransformerConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an AutoTransformer encoder."""

    type: str = schema_utils.StringOptions(
        ["auto_transformer"],
        default="auto_transformer",
        allow_none=False,
        description="Type of encoder.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlm-roberta-base",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["AutoTransformerEncoder"]["pretrained_model_name_or_path"],
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["AutoTransformerEncoder"]["max_sequence_length"],
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["AutoTransformerEncoder"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
        parameter_metadata=ENCODER_METADATA["AutoTransformerEncoder"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["AutoTransformerEncoder"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="Vocabulary size of the XLMRoBERTa model.",
        parameter_metadata=ENCODER_METADATA["AutoTransformerEncoder"]["vocab_size"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["AutoTransformerEncoder"]["pretrained_kwargs"],
    )
