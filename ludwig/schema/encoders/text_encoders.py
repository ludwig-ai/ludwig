from typing import Callable, List, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig


@dataclass
class ALBERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an ALBERT encoder."""

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
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
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
        min=0,
        max=1,
        description="The dropout ratio for attached classifiers.",
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="",
    )

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the token to use as padding.",
    )

    bos_token_id: int = schema_utils.Integer(
        default=2,
        description="The beginning of sequence token ID.",
    )

    eos_token_id: int = schema_utils.Integer(
        default=3,
        description="The end of sequence token ID.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class MT5Config(BaseEncoderConfig):
    """This dataclass configures the schema used for an MT5 encoder."""

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

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the token to use as padding.",
    )

    eos_token_id: int = schema_utils.Integer(
        default=1,
        description="The end of sequence token ID.",
    )

    decoder_start_token_id: int = schema_utils.Integer(
        default=0,
        description="If an encoder-decoder model starts decoding with a different token than _bos_, the id of that "
        "token.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class XLMRoBERTaConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an XLMRoBERTa encoder."""

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

    pad_token_id: int = schema_utils.Integer(
        default=1,
        description="The ID of the token to use as padding.",
    )

    bos_token_id: int = schema_utils.Integer(
        default=0,
        description="The beginning of sequence token ID.",
    )

    eos_token_id: int = schema_utils.Integer(
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
class BERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an BERT encoder."""

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

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(  # TODO: add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
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

    pad_token_id: int = schema_utils.Integer(
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
        min=0,
        max=1,
        description="The dropout ratio for the classification head.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class XLMConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an XLM encoder."""

    type: str = "xlm"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlm-mlm-en-2048",
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
        default=30145,
        description="Vocabulary size of the BERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling XLMModel or TFXLMModel.",
    )

    emb_dim: int = schema_utils.PositiveInteger(
        default=2048,
        description="Dimensionality of the encoder layers and the pooler layer.",
    )

    n_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
    )

    n_heads: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for the attention mechanism.",
    )

    gelu_activation: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use gelu for the activations instead of relu.",
    )

    sinusoidal_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.",
    )

    causal: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not the model should behave in a causal manner. Causal models use a triangular "
        "attention mask in order to only attend to the left-side context instead if a bidirectional "
        "context.",
    )

    asm: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the "
        "prediction layer.",
    )

    n_langs: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of languages the model handles. Set to 1 for monolingual models.",
    )

    use_lang_emb: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use language embeddings. Some models use additional language embeddings, "
        "see the multilingual models page for information on how to use them.",
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
    )

    embed_init_std: float = schema_utils.NonNegativeFloat(
        default=2048**-0.5,
        description="The standard deviation of the truncated_normal_initializer for initializing the embedding "
        "matrices.",
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
    )

    init_std: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices "
        "except the embedding matrices.",
    )

    bos_index: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The index of the beginning of sentence token in the vocabulary.",
    )

    eos_index: int = schema_utils.NonNegativeInteger(
        default=1,
        description="The index of the end of sentence token in the vocabulary.",
    )

    pad_index: int = schema_utils.NonNegativeInteger(
        default=2,
        description="The index of the padding token in the vocabulary.",
    )

    unk_index: int = schema_utils.NonNegativeInteger(
        default=3, description="The index of the unknown token in the vocabulary."
    )

    mask_index: int = schema_utils.NonNegativeInteger(
        default=5,
        description="The index of the masking token in the vocabulary.",
    )

    is_encoder: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not the initialized model should be a transformer encoder or decoder as seen in "
        "Vaswani et al.",
    )

    start_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description="Used in the SQuAD evaluation script.",
    )

    end_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description="Used in the SQuAD evaluation script.",
    )

    mask_token_id: int = schema_utils.Integer(
        default=0,
        description="Model agnostic parameter to identify masked tokens when generating text in an MLM context.",
    )

    lang_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the language used by the model. This parameter is used when generating text in a given "
        "language.",
    )

    pad_token_id: int = schema_utils.Integer(
        default=2,
        description="The ID of the token to use as padding.",
    )

    bos_token_id: int = schema_utils.Integer(
        default=0,
        description="The beginning of sequence token ID.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class GPTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an GPT encoder."""

    type: str = "gpt"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="openai-gpt",
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

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the GPT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling OpenAIGPTModel or TFOpenAIGPTModel.",
    )

    n_positions: int = schema_utils.PositiveInteger(
        default=40478,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
    )

    n_ctx: int = schema_utils.PositiveInteger(
        default=512,
        description="Dimensionality of the causal mask (usually same as n_positions)",
    )

    n_embd: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the embeddings and hidden states.",
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
    )

    n_head: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    afn: str = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu_new",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
    )

    resid_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    embd_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout ratio for the embeddings.",
    )

    attn_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout ratio for the attention.",
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-5,
        description="The epsilon to use in the layer normalization layers",
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class GPT2Config(BaseEncoderConfig):
    """This dataclass configures the schema used for an GPT2 encoder."""

    type: str = "gpt2"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="gpt2",
        description="Name or path of the pretrained model.",
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=50257,
        description="Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling GPT2Model or TFGPT2Model.",
    )

    n_positions: int = schema_utils.PositiveInteger(
        default=1024,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
    )

    n_ctx: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the causal mask (usually same as n_positions)",
    )

    n_embd: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the embeddings and hidden states.",
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
    )

    n_head: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    n_inner: int = schema_utils.PositiveInteger(
        default=None,
        description="Dimensionality of the inner feed-forward layers. None will set it to 4 times n_embd",
    )

    activation_function: str = schema_utils.StringOptions(
        ["relu", "silu", "gelu", "tanh", "gelu_new"],
        default="gelu",
        description="Activation function, to be selected in the list ['relu', 'silu', 'gelu', 'tanh', 'gelu_new'].",
    )

    resid_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    embd_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the embeddings.",
    )

    attn_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention.",
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-5,
        description="The epsilon to use in the layer normalization layers.",
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    scale_attn_weights: bool = schema_utils.Boolean(
        default=True,
        description="Scale attention weights by dividing by sqrt(hidden_size).",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class RoBERTaConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an RoBERTa encoder."""

    type: str = "roberta"

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="roberta-base",
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
        description="Vocabulary size of the RoBERTa model.",
    )

    pad_token_id: int = schema_utils.Integer(
        default=1,
        description="The ID of the token to use as padding.",
    )

    bos_token_id: int = schema_utils.Integer(
        default=0,
        description="The beginning of sequence token ID.",
    )

    eos_token_id: int = schema_utils.Integer(
        default=2,
        description="The end of sequence token ID.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class TransformerXLConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an TransformerXL encoder."""

    type: str = "transformer_xl"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="transfo-xl-wt103",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=267735,
        description="Vocabulary size of the TransfoXL model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling TransfoXLModel or TFTransfoXLModel.",
    )

    cutoffs: List[int] = schema_utils.List(
        int,
        default=[20000, 40000, 200000],
        description="Cutoffs for the adaptive softmax.",
    )

    d_model: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the model’s hidden states.",
    )

    d_embed: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the embeddings",
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    d_head: int = schema_utils.PositiveInteger(
        default=64,
        description="Dimensionality of the model’s heads.",
    )

    d_inner: int = schema_utils.PositiveInteger(
        default=4096,
        description=" Inner dimension in FF",
    )

    div_val: int = schema_utils.PositiveInteger(
        default=4,
        description="Divident value for adapative input and softmax.",
    )

    pre_lnorm: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to apply LayerNorm to the input instead of the output in the blocks.",
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=18,
        description="Number of hidden layers in the Transformer encoder.",
    )

    mem_len: int = schema_utils.PositiveInteger(
        default=1600,
        description="Length of the retained previous heads.",
    )

    clamp_len: int = schema_utils.PositiveInteger(
        default=1000,
        description="Use the same pos embeddings after clamp_len.",
    )

    same_length: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use the same attn length for all tokens",
    )

    proj_share_all_but_first: bool = schema_utils.Boolean(
        default=True,
        description="True to share all but first projs, False not to share.",
    )

    attn_type: int = schema_utils.IntegerRange(
        default=0,
        min=0,
        max=3,
        description="Attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.",
    )

    sample_softmax: int = schema_utils.Integer(
        default=-1,
        description="Number of samples in the sampled softmax.",
    )

    adaptive: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use adaptive softmax.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    dropatt: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="The dropout ratio for the attention probabilities.",
    )

    untie_r: bool = schema_utils.Boolean(
        default=True,
        description="Whether ot not to untie relative position biases.",
    )

    init: str = schema_utils.String(
        default="normal",
        description="Parameter initializer to use.",
    )

    init_range: float = schema_utils.NonNegativeFloat(
        default=0.01,
        description="Parameters initialized by U(-init_range, init_range).",
    )

    proj_init_std: float = schema_utils.NonNegativeFloat(
        default=0.01,
        description="Parameters initialized by N(0, init_std)",
    )

    init_std: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="Parameters initialized by N(0, init_std)",
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-5,
        description="The epsilon to use in the layer normalization layers",
    )

    eos_token_id: int = schema_utils.Integer(
        default=0,
        description="The end of sequence token ID.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class XLNetConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an XLNet encoder."""

    type: str = "xlnet"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlnet-base-cased",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=32000,
        description="Vocabulary size of the XLNet model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling XLNetModel or TFXLNetModel.",
    )

    d_model: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the encoder layers and the pooler layer.",
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=24,
        description="Number of hidden layers in the Transformer encoder.",
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    d_inner: int = schema_utils.PositiveInteger(
        default=4096,
        description="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
    )

    ff_activation: str = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler. If string, "
        "'gelu', 'relu', 'silu' and 'gelu_new' are supported.",
    )

    untie_r: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to untie relative position biases",
    )

    attn_type: str = schema_utils.StringOptions(
        ["bi", "uni"],
        default="bi",
        description="The attention type used by the model. Set 'bi' for XLNet, 'uni' for Transformer-XL.",
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    mem_len: int = schema_utils.PositiveInteger(
        default=512,
        description="The number of tokens to cache. The key/value pairs that have already been pre-computed in a "
        "previous forward pass won’t be re-computed. ",
    )

    reuse_len: int = schema_utils.PositiveInteger(
        default=None,
        description="The number of tokens in the current batch to be cached and reused in the future.",
    )

    use_mems_eval: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.",
    )

    use_mems_train: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not the model should make use of the recurrent memory mechanism in train mode.",
    )

    bi_data: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use bidirectional input pipeline. Usually set to True during pretraining and "
        "False during finetuning.",
    )

    clamp_len: int = schema_utils.Integer(
        default=-1,
        description="Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no "
        "clamping.",
    )

    same_length: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use the same attention length for each token.",
    )

    summary_type: str = schema_utils.StringOptions(
        ["last", "first", "mean", "cls_index", "attn"],
        default="last",
        description="Argument used when doing sequence summary. Used in the sequence classification and multiple "
        "choice models.",
    )

    summary_use_proj: bool = schema_utils.Boolean(
        default=True,
        description="",
    )

    summary_activation: str = schema_utils.String(
        default="tanh",
        description="Argument used when doing sequence summary. Used in the sequence classification and multiple "
        "choice models.",
    )

    summary_last_dropout: float = schema_utils.FloatRange(
        default=0.1,
        description="Used in the sequence classification and multiple choice models.",
    )

    start_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description="Used in the SQuAD evaluation script.",
    )

    end_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description=" Used in the SQuAD evaluation script.",
    )

    pad_token_id: int = schema_utils.Integer(
        default=5,
        description="The ID of the token to use as padding.",
    )

    bos_token_id: int = schema_utils.Integer(
        default=1,
        description="The beginning of sequence token ID.",
    )

    eos_token_id: int = schema_utils.Integer(
        default=2,
        description="The end of sequence token ID.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class DistilBERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an DistilBERT encoder."""

    type: str = "distilbert"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="distilbert-base-uncased",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether to train the model.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling DistilBertModel or TFDistilBertModel.",
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
    )

    sinusoidal_pos_embds: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use sinusoidal positional embeddings.",
    )

    n_layers: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of hidden layers in the Transformer encoder.",
    )

    n_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
    )

    dim: int = schema_utils.PositiveInteger(
        default=768,
        description=" Dimensionality of the encoder layers and the pooler layer.",
    )

    hidden_dim: int = schema_utils.PositiveInteger(
        default=3072,
        description="The size of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_dropout: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the attention probabilities.",
    )

    activation: Union[str, Callable] = schema_utils.StringOptions(  # TODO: Add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler. If string, "
        "'gelu', 'relu', 'silu' and 'gelu_new' are supported.",
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    qa_dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probabilities used in the question answering model DistilBertForQuestionAnswering.",
    )

    seq_classif_dropout: float = schema_utils.FloatRange(
        default=0.2,
        min=0,
        max=1,
        description="The dropout probabilities used in the sequence classification and the multiple choice model "
        "DistilBertForSequenceClassification.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class CTRLConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an CTRL encoder."""

    type: str = "ctrl"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="ctrl",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=246534,
        description="Vocabulary size of the CTRL model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling CTRLModel or TFCTRLModel.",
    )

    n_positions: int = schema_utils.PositiveInteger(
        default=256,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
    )

    n_ctx: int = schema_utils.PositiveInteger(
        default=256,
        description="Dimensionality of the causal mask (usually same as n_positions)",
    )

    n_embd: int = schema_utils.PositiveInteger(
        default=1280,
        description="Dimensionality of the embeddings and hidden states.",
    )

    dff: int = schema_utils.PositiveInteger(
        default=8192,
        description="Dimensionality of the inner dimension of the feed forward networks (FFN).",
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=48,
        description="Number of hidden layers in the Transformer encoder.",
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    resid_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description=" The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    embd_pdrop: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the embeddings.",
    )

    attn_pdrop: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the attention.",
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-6,
        description="The epsilon to use in the layer normalization layers",
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class CamemBERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an CamemBERT encoder."""

    type: str = "camembert"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="jplu/camembert-base",
        description="Name or path of the pretrained model.",
    )

    reduce_output: str = schema_utils.String(
        default="cls-pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the CamemBERT model.",
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

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(  # TODO: add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
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

    pad_token_id: int = schema_utils.Integer(
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
        min=0,
        max=1,
        description="The dropout ratio for the classification head.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class T5Config(BaseEncoderConfig):
    """This dataclass configures the schema used for an T5 encoder."""

    type: str = "t5"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="t5-small",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=32128,
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
        default=2048,
        description="Size of the intermediate feed forward layer in each T5Block.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of hidden layers in the Transformer encoder.",
    )

    num_decoder_layers: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of hidden layers in the Transformer decoder. Will use the same value as num_layers if not "
        "set.",
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    relative_attention_num_buckets: int = schema_utils.PositiveInteger(
        default=32,
        description="The number of buckets to use for each attention layer.",
    )

    dropout_rate: float = schema_utils.FloatRange(
        default=0.1,
        description="The ratio for all dropout layers.",
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-6,
        description="The epsilon used by the layer normalization layers.",
    )

    initializer_factor: float = schema_utils.NonNegativeFloat(
        default=1,
        description="A factor for initializing all weight matrices (should be kept to 1, used internally for "
        "initialization testing).",
    )

    feed_forward_proj: str = schema_utils.StringOptions(
        ["relu", "gated-gelu"],
        default="relu",
        description="Type of feed forward layer to be used. Should be one of 'relu' or 'gated-gelu'. T5v1.1 uses the "
        "'gated-gelu' feed forward projection. Original T5 uses 'relu'.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class FlauBERTConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an FlauBERT encoder."""

    type: str = "flaubert"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="t5-small",
        description="flaubert/flaubert_small_cased",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30145,
        description="Vocabulary size of the FlauBERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling FlaubertModel or TFFlaubertModel.",
    )

    pre_norm: bool = schema_utils.Boolean(
        default=False,
        description="Whether to apply the layer normalization before or after the feed forward layer following the "
        "attention in each layer (Vaswani et al., Tensor2Tensor for Neural Machine Translation. 2018)",
    )

    layerdrop: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Probability to drop layers during training (Fan et al., Reducing Transformer Depth on Demand "
        "with Structured Dropout. ICLR 2020)",
    )

    emb_dim: int = schema_utils.PositiveInteger(
        default=2048,
        description="Dimensionality of the encoder layers and the pooler layer.",
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for the attention mechanism",
    )

    gelu_activation: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use a gelu activation instead of relu.",
    )

    sinusoidal_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.",
    )

    causal: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not the model should behave in a causal manner. Causal models use a triangular "
        "attention mask in order to only attend to the left-side context instead if a bidirectional "
        "context.",
    )

    asm: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the "
        "prediction layer.",
    )

    n_langs: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of languages the model handles. Set to 1 for monolingual models.",
    )

    use_lang_emb: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use language embeddings. Some models use additional language embeddings, "
        "see the multilingual models page for information on how to use them.",
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
    )

    embed_init_std: float = schema_utils.NonNegativeFloat(
        default=2048**-0.5,
        description="The standard deviation of the truncated_normal_initializer for initializing the embedding "
        "matrices.",
    )

    init_std: int = schema_utils.PositiveInteger(
        default=50257,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices "
        "except the embedding matrices.",
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
    )

    bos_index: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The index of the beginning of sentence token in the vocabulary.",
    )

    eos_index: int = schema_utils.NonNegativeInteger(
        default=1,
        description="The index of the end of sentence token in the vocabulary.",
    )

    pad_index: int = schema_utils.NonNegativeInteger(
        default=2,
        description="The index of the padding token in the vocabulary.",
    )

    unk_index: int = schema_utils.NonNegativeInteger(
        default=3,
        description="The index of the unknown token in the vocabulary.",
    )

    mask_index: int = schema_utils.NonNegativeInteger(
        default=5,
        description="The index of the masking token in the vocabulary.",
    )

    is_encoder: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not the initialized model should be a transformer encoder or decoder as seen in "
        "Vaswani et al.",
    )

    mask_token_id: int = schema_utils.Integer(
        default=0,
        description="Model agnostic parameter to identify masked tokens when generating text in an MLM context.",
    )

    lang_id: int = schema_utils.Integer(
        default=1,
        description="The ID of the language used by the model. This parameter is used when generating text in a given "
        "language.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class ELECTRAConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an ELECTRA encoder."""

    type: str = "electra"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="google/electra-small-discriminator",
        description="Name or path of the pretrained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to train the model.",
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the ELECTRA model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling ElectraModel or TFElectraModel.",
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Dimensionality of the encoder layers and the pooler layer.",
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Dimensionality of the encoder layers and the pooler layer.",
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=4,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.",
    )

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(  # TODO: add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention probabilities.",
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling ElectraModel or TFElectraModel.",
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="Type of position embedding.",
    )

    classifier_dropout: float = schema_utils.FloatRange(
        default=None,
        min=0,
        max=1,
        description="The dropout ratio for the classification head.",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class LongformerConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an Longformer encoder."""

    type: str = "longformer"

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        description="Maximum length of the input sequence.",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model.",
    )

    attention_window: Union[List[int], int] = schema_utils.IntegerOrSequenceOfIntegers(
        default=512,
        description="Size of an attention window around each token. If an int, use the same size for all layers. To "
        "specify a different window size for each layer, use a List[int] where len(attention_window) == "
        "num_hidden_layers.",
    )

    sep_token_id: int = schema_utils.Integer(
        default=2,
        description="ID of the separator token, which is used when building a sequence from multiple sequences",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="allenai/longformer-base-4096",
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

    num_tokens: int = schema_utils.PositiveInteger(
        default=None,
        description="Number of tokens",
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
    )


@dataclass
class AutoTransformerConfig(BaseEncoderConfig):
    """This dataclass configures the schema used for an AutoTransformer encoder."""

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
