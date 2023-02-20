from typing import Any, Callable, Dict, List, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TEXT
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.sequence_encoders import SequenceEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.schema.metadata.parameter_metadata import ParameterMetadata
from ludwig.schema.utils import ludwig_dataclass


class HFEncoderConfig(SequenceEncoderConfig):
    trainable: bool
    use_pretrained: bool
    pretrained_model_name_or_path: str
    reduce_output: str

    def get_fixed_preprocessing_params(self) -> Dict[str, Any]:
        model_name = self.pretrained_model_name_or_path
        if model_name is None and self.use_pretrained:
            # no default model name, so model name is required by the subclass
            raise ValueError(
                f"Missing required parameter for `{self.type}` encoder: `pretrained_model_name_or_path` when "
                "`use_pretrained` is True."
            )
        params = {
            "tokenizer": "hf_tokenizer",
            "pretrained_model_name_or_path": model_name,
        }

        if not self.can_cache_embeddings():
            params["cache_encoder_embeddings"] = False

        return params

    def is_pretrained(self) -> bool:
        return self.use_pretrained

    def can_cache_embeddings(self) -> bool:
        """Returns true if the encoder's output embeddings will not change during training."""
        return not self.trainable and self.reduce_output != "attention"


@DeveloperAPI
@register_encoder_config("albert", TEXT)
@ludwig_dataclass
class ALBERTConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an ALBERT encoder."""

    @staticmethod
    def module_name():
        return "ALBERT"

    type: str = schema_utils.ProtectedString(
        "albert",
        description=ENCODER_METADATA["ALBERT"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="albert-base-v2",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["trainable"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["reduce_output"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30000,
        description="Vocabulary size of the ALBERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["vocab_size"],
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Dimensionality of vocabulary embeddings.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["embedding_size"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["hidden_size"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["num_hidden_layers"],
    )

    num_hidden_groups: int = schema_utils.PositiveInteger(
        default=1,
        description="Number of groups for the hidden layers, parameters in the same group are shared.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["num_hidden_groups"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["num_attention_heads"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=3072,
        description="The dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer "
        "encoder.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["intermediate_size"],
    )

    inner_group_num: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of inner repetition of attention and ffn.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["inner_group_num"],
    )

    hidden_act: str = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu_new",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["hidden_dropout_prob"],
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["attention_probs_dropout_prob"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling AlbertModel or TFAlbertModel.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["type_vocab_size"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["layer_norm_eps"],
    )

    classifier_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for attached classifiers.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["classifier_dropout_prob"],
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["position_embedding_type"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=2,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["bos_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=3,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["eos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["ALBERT"]["pretrained_kwargs"],
    )


# TODO: uncomment when sentencepiece doesn't cause segfaults: https://github.com/ludwig-ai/ludwig/issues/2983
@DeveloperAPI
# @register_encoder_config("mt5", TEXT)
@ludwig_dataclass
class MT5Config(HFEncoderConfig):
    """This dataclass configures the schema used for an MT5 encoder."""

    @staticmethod
    def module_name():
        return "MT5"

    type: str = schema_utils.ProtectedString(
        "mt5",
        description=ENCODER_METADATA["MT5"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["MT5"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["MT5"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="google/mt5-base",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["MT5"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["MT5"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["MT5"]["trainable"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["MT5"]["reduce_output"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["MT5"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=250112,
        description="Vocabulary size of the T5 model. Defines the number of different tokens that can be represented "
        "by the inputs_ids passed when calling T5Model or TFT5Model.",
        parameter_metadata=ENCODER_METADATA["MT5"]["vocab_size"],
    )

    d_model: int = schema_utils.PositiveInteger(
        default=512,
        description="Size of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["MT5"]["d_model"],
    )

    d_kv: int = schema_utils.PositiveInteger(
        default=64,
        description="Size of the key, query, value projections per attention head. d_kv has to be equal to d_model // "
        "num_heads.",
        parameter_metadata=ENCODER_METADATA["MT5"]["d_kv"],
    )

    d_ff: int = schema_utils.PositiveInteger(
        default=1024,
        description="Size of the intermediate feed forward layer in each T5Block.",
        parameter_metadata=ENCODER_METADATA["MT5"]["d_ff"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["MT5"]["num_layers"],
    )

    num_decoder_layers: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of hidden layers in the Transformer decoder. Will use the same value as num_layers if not "
        "set.",
        parameter_metadata=ENCODER_METADATA["MT5"]["num_decoder_layers"],
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["MT5"]["num_heads"],
    )

    relative_attention_num_buckets: int = schema_utils.PositiveInteger(
        default=32,
        description="The number of buckets to use for each attention layer.",
        parameter_metadata=ENCODER_METADATA["MT5"]["relative_attention_num_buckets"],
    )

    dropout_rate: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The ratio for all dropout layers.",
        parameter_metadata=ENCODER_METADATA["MT5"]["dropout_rate"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-06,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["MT5"]["layer_norm_epsilon"],
    )

    initializer_factor: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="A factor for initializing all weight matrices (should be kept to 1, used internally for "
        "initialization testing)",
        parameter_metadata=ENCODER_METADATA["MT5"]["initializer_factor"],
    )

    feed_forward_proj: str = schema_utils.StringOptions(
        ["relu", "gated-gelu"],
        default="gated-gelu",
        description="Type of feed forward layer to be used. ",
        parameter_metadata=ENCODER_METADATA["MT5"]["feed_forward_proj"],
    )

    is_encoder_decoder: bool = schema_utils.Boolean(
        default=True,
        description="",
        parameter_metadata=ENCODER_METADATA["MT5"]["is_encoder_decoder"],
    )

    use_cache: bool = schema_utils.Boolean(
        default=True,
        description="",
        parameter_metadata=ENCODER_METADATA["MT5"]["use_cache"],
    )

    tokenizer_class: str = schema_utils.String(
        default="T5Tokenizer",
        description="",
        parameter_metadata=ENCODER_METADATA["MT5"]["tokenizer_class"],
    )

    tie_word_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Whether the model's input and output word embeddings should be tied.",
        parameter_metadata=ENCODER_METADATA["MT5"]["tie_word_embeddings"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["MT5"]["pad_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=1,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["MT5"]["eos_token_id"],
    )

    decoder_start_token_id: int = schema_utils.Integer(
        default=0,
        description="If an encoder-decoder model starts decoding with a different token than _bos_, the id of that "
        "token.",
        parameter_metadata=ENCODER_METADATA["MT5"]["decoder_start_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["MT5"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("xlmroberta", TEXT)
@ludwig_dataclass
class XLMRoBERTaConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an XLMRoBERTa encoder."""

    @staticmethod
    def module_name():
        return "XLMRoBERTa"

    type: str = schema_utils.ProtectedString(
        "xlmroberta",
        description=ENCODER_METADATA["XLMRoBERTa"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlm-roberta-base",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Vocabulary size of the XLMRoBERTa model.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["vocab_size"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=1,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=0,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["bos_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=2,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["eos_token_id"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=514,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=1,
        description="The vocabulary size of the token_type_ids passed in.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["type_vocab_size"],
    )

    add_pooling_layer: bool = schema_utils.Boolean(
        default=True,
        description="Whether to add a pooling layer to the encoder.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["add_pooling_layer"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLMRoBERTa"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("bert", TEXT)
@ludwig_dataclass
class BERTConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an BERT encoder."""

    @staticmethod
    def module_name():
        return "BERT"

    type: str = schema_utils.ProtectedString(
        "bert",
        description=ENCODER_METADATA["BERT"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["BERT"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["BERT"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="bert-base-uncased",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["BERT"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["BERT"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["BERT"]["trainable"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["BERT"]["reduce_output"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["BERT"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the BERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling BertModel or TFBertModel.",
        parameter_metadata=ENCODER_METADATA["BERT"]["vocab_size"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["BERT"]["hidden_size"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["BERT"]["num_hidden_layers"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["BERT"]["num_attention_heads"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=3072,
        description="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["BERT"]["intermediate_size"],
    )

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(  # TODO: add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["BERT"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["BERT"]["hidden_dropout_prob"],
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["BERT"]["attention_probs_dropout_prob"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["BERT"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel.",
        parameter_metadata=ENCODER_METADATA["BERT"]["type_vocab_size"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["BERT"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["BERT"]["layer_norm_eps"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["BERT"]["pad_token_id"],
    )

    gradient_checkpointing: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use gradient checkpointing.",
        parameter_metadata=ENCODER_METADATA["BERT"]["gradient_checkpointing"],
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="Type of position embedding.",
        parameter_metadata=ENCODER_METADATA["BERT"]["position_embedding_type"],
    )

    classifier_dropout: float = schema_utils.FloatRange(
        default=None,
        allow_none=True,
        min=0,
        max=1,
        description="The dropout ratio for the classification head.",
        parameter_metadata=ENCODER_METADATA["BERT"]["classifier_dropout"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["BERT"]["pretrained_kwargs"],
    )


# TODO: uncomment once we figure out host memory issue: https://github.com/ludwig-ai/ludwig/issues/3107
@DeveloperAPI
# @register_encoder_config("xlm", TEXT)
@ludwig_dataclass
class XLMConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an XLM encoder."""

    @staticmethod
    def module_name():
        return "XLM"

    type: str = schema_utils.ProtectedString(
        "xlm",
        description=ENCODER_METADATA["XLM"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["XLM"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["XLM"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlm-mlm-en-2048",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLM"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["XLM"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["XLM"]["trainable"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["XLM"]["reduce_output"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["XLM"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30145,
        description="Vocabulary size of the BERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling XLMModel or TFXLMModel.",
        parameter_metadata=ENCODER_METADATA["XLM"]["vocab_size"],
    )

    emb_dim: int = schema_utils.PositiveInteger(
        default=2048,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["XLM"]["emb_dim"],
    )

    n_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLM"]["n_layers"],
    )

    n_heads: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLM"]["n_heads"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["XLM"]["dropout"],
    )

    attention_dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for the attention mechanism.",
        parameter_metadata=ENCODER_METADATA["XLM"]["attention_dropout"],
    )

    gelu_activation: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use gelu for the activations instead of relu.",
        parameter_metadata=ENCODER_METADATA["XLM"]["gelu_activation"],
    )

    sinusoidal_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.",
        parameter_metadata=ENCODER_METADATA["XLM"]["sinusoidal_embeddings"],
    )

    causal: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not the model should behave in a causal manner. Causal models use a triangular "
        "attention mask in order to only attend to the left-side context instead if a bidirectional "
        "context.",
        parameter_metadata=ENCODER_METADATA["XLM"]["causal"],
    )

    asm: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the "
        "prediction layer.",
        parameter_metadata=ENCODER_METADATA["XLM"]["asm"],
    )

    n_langs: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of languages the model handles. Set to 1 for monolingual models.",
        parameter_metadata=ENCODER_METADATA["XLM"]["n_langs"],
    )

    use_lang_emb: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use language embeddings. Some models use additional language embeddings, "
        "see the multilingual models page for information on how to use them.",
        parameter_metadata=ENCODER_METADATA["XLM"]["use_lang_emb"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["XLM"]["max_position_embeddings"],
    )

    embed_init_std: float = schema_utils.NonNegativeFloat(
        default=2048**-0.5,
        description="The standard deviation of the truncated_normal_initializer for initializing the embedding "
        "matrices.",
        parameter_metadata=ENCODER_METADATA["XLM"]["embed_init_std"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["XLM"]["layer_norm_eps"],
    )

    init_std: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices "
        "except the embedding matrices.",
        parameter_metadata=ENCODER_METADATA["XLM"]["init_std"],
    )

    bos_index: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The index of the beginning of sentence token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLM"]["bos_index"],
    )

    eos_index: int = schema_utils.NonNegativeInteger(
        default=1,
        description="The index of the end of sentence token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLM"]["eos_index"],
    )

    pad_index: int = schema_utils.NonNegativeInteger(
        default=2,
        description="The index of the padding token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLM"]["pad_index"],
    )

    unk_index: int = schema_utils.NonNegativeInteger(
        default=3,
        description="The index of the unknown token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLM"]["unk_index"],
    )

    mask_index: int = schema_utils.NonNegativeInteger(
        default=5,
        description="The index of the masking token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["XLM"]["mask_index"],
    )

    is_encoder: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not the initialized model should be a transformer encoder or decoder as seen in "
        "Vaswani et al.",
        parameter_metadata=ENCODER_METADATA["XLM"]["is_encoder"],
    )

    start_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description="Used in the SQuAD evaluation script.",
        parameter_metadata=ENCODER_METADATA["XLM"]["start_n_top"],
    )

    end_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description="Used in the SQuAD evaluation script.",
        parameter_metadata=ENCODER_METADATA["XLM"]["end_n_top"],
    )

    mask_token_id: int = schema_utils.Integer(
        default=0,
        description="Model agnostic parameter to identify masked tokens when generating text in an MLM context.",
        parameter_metadata=ENCODER_METADATA["XLM"]["mask_token_id"],
    )

    lang_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the language used by the model. This parameter is used when generating text in a given "
        "language.",
        parameter_metadata=ENCODER_METADATA["XLM"]["lang_id"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=2,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["XLM"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=0,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLM"]["bos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLM"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("gpt", TEXT)
@ludwig_dataclass
class GPTConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an GPT encoder."""

    @staticmethod
    def module_name():
        return "GPT"

    type: str = schema_utils.ProtectedString(
        "gpt",
        description=ENCODER_METADATA["GPT"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["GPT"]["max_sequence_length"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["GPT"]["reduce_output"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["GPT"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="openai-gpt",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["GPT"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["GPT"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["GPT"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["GPT"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the GPT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling OpenAIGPTModel or TFOpenAIGPTModel.",
        parameter_metadata=ENCODER_METADATA["GPT"]["vocab_size"],
    )

    n_positions: int = schema_utils.PositiveInteger(
        default=40478,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["GPT"]["n_positions"],
    )

    n_ctx: int = schema_utils.PositiveInteger(
        default=512,
        description="Dimensionality of the causal mask (usually same as n_positions)",
        parameter_metadata=ENCODER_METADATA["GPT"]["n_ctx"],
    )

    n_embd: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the embeddings and hidden states.",
        parameter_metadata=ENCODER_METADATA["GPT"]["n_embd"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["GPT"]["n_layer"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["GPT"]["n_head"],
    )

    afn: str = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu_new",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["GPT"]["afn"],
    )

    resid_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["GPT"]["resid_pdrop"],
    )

    embd_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout ratio for the embeddings.",
        parameter_metadata=ENCODER_METADATA["GPT"]["embd_pdrop"],
    )

    attn_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout ratio for the attention.",
        parameter_metadata=ENCODER_METADATA["GPT"]["attn_pdrop"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-5,
        description="The epsilon to use in the layer normalization layers",
        parameter_metadata=ENCODER_METADATA["GPT"]["layer_norm_epsilon"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["GPT"]["initializer_range"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["GPT"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("gpt2", TEXT)
@ludwig_dataclass
class GPT2Config(HFEncoderConfig):
    """This dataclass configures the schema used for an GPT2 encoder."""

    @staticmethod
    def module_name():
        return "GPT2"

    type: str = schema_utils.ProtectedString(
        "gpt2",
        description=ENCODER_METADATA["GPT2"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="gpt2",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["pretrained_model_name_or_path"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["GPT2"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=50257,
        description="Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling GPT2Model or TFGPT2Model.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["vocab_size"],
    )

    n_positions: int = schema_utils.PositiveInteger(
        default=1024,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["GPT2"]["n_positions"],
    )

    n_ctx: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the causal mask (usually same as n_positions)",
        parameter_metadata=ENCODER_METADATA["GPT2"]["n_ctx"],
    )

    n_embd: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the embeddings and hidden states.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["n_embd"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["n_layer"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["n_head"],
    )

    n_inner: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Dimensionality of the inner feed-forward layers. None will set it to 4 times n_embd",
        parameter_metadata=ENCODER_METADATA["GPT2"]["n_inner"],
    )

    activation_function: str = schema_utils.StringOptions(
        ["relu", "silu", "gelu", "tanh", "gelu_new"],
        default="gelu_new",
        description="Activation function, to be selected in the list ['relu', 'silu', 'gelu', 'tanh', 'gelu_new'].",
        parameter_metadata=ENCODER_METADATA["GPT2"]["activation_function"],
    )

    resid_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["resid_pdrop"],
    )

    embd_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the embeddings.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["embd_pdrop"],
    )

    attn_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["attn_pdrop"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-5,
        description="The epsilon to use in the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["layer_norm_epsilon"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["initializer_range"],
    )

    scale_attn_weights: bool = schema_utils.Boolean(
        default=True,
        description="Scale attention weights by dividing by sqrt(hidden_size).",
        parameter_metadata=ENCODER_METADATA["GPT2"]["scale_attn_weights"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["GPT2"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("roberta", TEXT)
@ludwig_dataclass
class RoBERTaConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an RoBERTa encoder."""

    @staticmethod
    def module_name():
        return "RoBERTa"

    type: str = schema_utils.ProtectedString(
        "roberta",
        description=ENCODER_METADATA["RoBERTa"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="roberta-base",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="cls_pooled",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Vocabulary size of the RoBERTa model.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["vocab_size"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=1,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=0,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["bos_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=2,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["eos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["RoBERTa"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("transformer_xl", TEXT)
@ludwig_dataclass
class TransformerXLConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an TransformerXL encoder."""

    @staticmethod
    def module_name():
        return "TransformerXL"

    type: str = schema_utils.ProtectedString(
        "transformer_xl",
        description=ENCODER_METADATA["TransformerXL"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="transfo-xl-wt103",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=267735,
        description="Vocabulary size of the TransfoXL model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling TransfoXLModel or TFTransfoXLModel.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["vocab_size"],
    )

    cutoffs: List[int] = schema_utils.List(
        int,
        default=[20000, 40000, 200000],
        description="Cutoffs for the adaptive softmax.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["cutoffs"],
    )

    d_model: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the model’s hidden states.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["d_model"],
    )

    d_embed: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the embeddings",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["d_embed"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["n_head"],
    )

    d_head: int = schema_utils.PositiveInteger(
        default=64,
        description="Dimensionality of the model’s heads.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["d_head"],
    )

    d_inner: int = schema_utils.PositiveInteger(
        default=4096,
        description=" Inner dimension in FF",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["d_inner"],
    )

    div_val: int = schema_utils.PositiveInteger(
        default=4,
        description="Divident value for adapative input and softmax.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["div_val"],
    )

    pre_lnorm: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to apply LayerNorm to the input instead of the output in the blocks.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["pre_lnorm"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=18,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["n_layer"],
    )

    mem_len: int = schema_utils.PositiveInteger(
        default=1600,
        description="Length of the retained previous heads.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["mem_len"],
    )

    clamp_len: int = schema_utils.PositiveInteger(
        default=1000,
        description="Use the same pos embeddings after clamp_len.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["clamp_len"],
    )

    same_length: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use the same attn length for all tokens",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["same_length"],
    )

    proj_share_all_but_first: bool = schema_utils.Boolean(
        default=True,
        description="True to share all but first projs, False not to share.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["proj_share_all_but_first"],
    )

    attn_type: int = schema_utils.IntegerRange(
        default=0,
        min=0,
        max=3,
        description="Attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["attn_type"],
    )

    sample_softmax: int = schema_utils.Integer(
        default=-1,
        description="Number of samples in the sampled softmax.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["sample_softmax"],
    )

    adaptive: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use adaptive softmax.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["adaptive"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["dropout"],
    )

    dropatt: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["dropatt"],
    )

    untie_r: bool = schema_utils.Boolean(
        default=True,
        description="Whether ot not to untie relative position biases.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["untie_r"],
    )

    init: str = schema_utils.String(
        default="normal",
        description="Parameter initializer to use.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["init"],
    )

    init_range: float = schema_utils.NonNegativeFloat(
        default=0.01,
        description="Parameters initialized by U(-init_range, init_range).",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["init_range"],
    )

    proj_init_std: float = schema_utils.NonNegativeFloat(
        default=0.01,
        description="Parameters initialized by N(0, init_std)",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["proj_init_std"],
    )

    init_std: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="Parameters initialized by N(0, init_std)",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["init_std"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-5,
        description="The epsilon to use in the layer normalization layers",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["layer_norm_epsilon"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=0,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["eos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["TransformerXL"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("xlnet", TEXT)
@ludwig_dataclass
class XLNetConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an XLNet encoder."""

    @staticmethod
    def module_name():
        return "XLNet"

    type: str = schema_utils.ProtectedString(
        "xlnet",
        description=ENCODER_METADATA["XLNet"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="xlnet-base-cased",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["XLNet"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=32000,
        description="Vocabulary size of the XLNet model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling XLNetModel or TFXLNetModel.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["vocab_size"],
    )

    d_model: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["d_model"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["n_layer"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["n_head"],
    )

    d_inner: int = schema_utils.PositiveInteger(
        default=3072,
        description="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["d_inner"],
    )

    ff_activation: str = schema_utils.StringOptions(
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler. If string, "
        "'gelu', 'relu', 'silu' and 'gelu_new' are supported.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["ff_activation"],
    )

    untie_r: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to untie relative position biases",
        parameter_metadata=ENCODER_METADATA["XLNet"]["untie_r"],
    )

    attn_type: str = schema_utils.StringOptions(
        ["bi"],
        default="bi",
        description="The attention type used by the model. Currently only 'bi' is supported.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["attn_type"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["layer_norm_eps"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["dropout"],
    )

    mem_len: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The number of tokens to cache. The key/value pairs that have already been pre-computed in a "
        "previous forward pass won’t be re-computed. ",
        parameter_metadata=ENCODER_METADATA["XLNet"]["mem_len"],
    )

    reuse_len: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The number of tokens in the current batch to be cached and reused in the future.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["reuse_len"],
    )

    use_mems_eval: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["use_mems_eval"],
    )

    use_mems_train: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not the model should make use of the recurrent memory mechanism in train mode.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["use_mems_train"],
    )

    bi_data: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use bidirectional input pipeline. Usually set to True during pretraining and "
        "False during finetuning.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["bi_data"],
    )

    clamp_len: int = schema_utils.Integer(
        default=-1,
        description="Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no "
        "clamping.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["clamp_len"],
    )

    same_length: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use the same attention length for each token.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["same_length"],
    )

    summary_type: str = schema_utils.StringOptions(
        ["last", "first", "mean", "cls_index", "attn"],
        default="last",
        description="Argument used when doing sequence summary. Used in the sequence classification and multiple "
        "choice models.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["summary_type"],
    )

    summary_use_proj: bool = schema_utils.Boolean(
        default=True,
        description="",
        parameter_metadata=ENCODER_METADATA["XLNet"]["summary_use_proj"],
    )

    summary_activation: str = schema_utils.String(
        default="tanh",
        description="Argument used when doing sequence summary. Used in the sequence classification and multiple "
        "choice models.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["summary_activation"],
    )

    summary_last_dropout: float = schema_utils.FloatRange(
        default=0.1,
        description="Used in the sequence classification and multiple choice models.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["summary_last_dropout"],
    )

    start_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description="Used in the SQuAD evaluation script.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["start_n_top"],
    )

    end_n_top: int = schema_utils.PositiveInteger(
        default=5,
        description=" Used in the SQuAD evaluation script.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["end_n_top"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=5,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["pad_token_id"],
    )

    bos_token_id: int = schema_utils.Integer(
        default=1,
        description="The beginning of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["bos_token_id"],
    )

    eos_token_id: int = schema_utils.Integer(
        default=2,
        description="The end of sequence token ID.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["eos_token_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["XLNet"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("distilbert", TEXT)
@ludwig_dataclass
class DistilBERTConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an DistilBERT encoder."""

    @staticmethod
    def module_name():
        return "DistilBERT"

    type: str = schema_utils.ProtectedString(
        "distilbert",
        description=ENCODER_METADATA["DistilBERT"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="distilbert-base-uncased",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling DistilBertModel or TFDistilBertModel.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["vocab_size"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["dropout"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["max_position_embeddings"],
    )

    sinusoidal_pos_embds: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use sinusoidal positional embeddings.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["sinusoidal_pos_embds"],
    )

    n_layers: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["n_layers"],
    )

    n_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["n_heads"],
    )

    dim: int = schema_utils.PositiveInteger(
        default=768,
        description=" Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["dim"],
    )

    hidden_dim: int = schema_utils.PositiveInteger(
        default=3072,
        description="The size of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["hidden_dim"],
    )

    attention_dropout: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["attention_dropout"],
    )

    activation: Union[str, Callable] = schema_utils.StringOptions(  # TODO: Add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler. If string, "
        "'gelu', 'relu', 'silu' and 'gelu_new' are supported.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["activation"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["initializer_range"],
    )

    qa_dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probabilities used in the question answering model DistilBertForQuestionAnswering.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["qa_dropout"],
    )

    seq_classif_dropout: float = schema_utils.FloatRange(
        default=0.2,
        min=0,
        max=1,
        description="The dropout probabilities used in the sequence classification and the multiple choice model "
        "DistilBertForSequenceClassification.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["seq_classif_dropout"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["DistilBERT"]["pretrained_kwargs"],
    )


# TODO: uncomment when CTRL bug (https://github.com/ludwig-ai/ludwig/issues/2977) has been fixed to add back in
@DeveloperAPI
# @register_encoder_config("ctrl", TEXT)
@ludwig_dataclass
class CTRLConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an CTRL encoder."""

    @staticmethod
    def module_name():
        return "CTRL"

    type: str = schema_utils.ProtectedString(
        "ctrl",
        description=ENCODER_METADATA["CTRL"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="ctrl",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["CTRL"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=246534,
        description="Vocabulary size of the CTRL model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling CTRLModel or TFCTRLModel.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["vocab_size"],
    )

    n_positions: int = schema_utils.PositiveInteger(
        default=256,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["CTRL"]["n_positions"],
    )

    n_ctx: int = schema_utils.PositiveInteger(
        default=256,
        description="Dimensionality of the causal mask (usually same as n_positions)",
        parameter_metadata=ENCODER_METADATA["CTRL"]["n_ctx"],
    )

    n_embd: int = schema_utils.PositiveInteger(
        default=1280,
        description="Dimensionality of the embeddings and hidden states.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["n_embd"],
    )

    dff: int = schema_utils.PositiveInteger(
        default=8192,
        description="Dimensionality of the inner dimension of the feed forward networks (FFN).",
        parameter_metadata=ENCODER_METADATA["CTRL"]["dff"],
    )

    n_layer: int = schema_utils.PositiveInteger(
        default=48,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["n_layer"],
    )

    n_head: int = schema_utils.PositiveInteger(
        default=16,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["n_head"],
    )

    resid_pdrop: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description=" The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["resid_pdrop"],
    )

    embd_pdrop: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the embeddings.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["embd_pdrop"],
    )

    attn_pdrop: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the attention.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["attn_pdrop"],
    )

    layer_norm_epsilon: float = schema_utils.NonNegativeFloat(
        default=1e-6,
        description="The epsilon to use in the layer normalization layers",
        parameter_metadata=ENCODER_METADATA["CTRL"]["layer_norm_epsilon"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["initializer_range"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["CTRL"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("camembert", TEXT)
@ludwig_dataclass
class CamemBERTConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an CamemBERT encoder."""

    @staticmethod
    def module_name():
        return "CamemBERT"

    type: str = schema_utils.ProtectedString(
        "camembert",
        description=ENCODER_METADATA["CamemBERT"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["use_pretrained"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["saved_weights_in_checkpoint"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="camembert-base",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["pretrained_model_name_or_path"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=32005,
        description="Vocabulary size of the CamemBERT model.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["vocab_size"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=768,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["hidden_size"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["num_hidden_layers"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["num_attention_heads"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=3072,
        description="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["intermediate_size"],
    )

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(  # TODO: add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["hidden_dropout_prob"],
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["attention_probs_dropout_prob"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=514,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=1,
        description="The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["type_vocab_size"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-05,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["layer_norm_eps"],
    )

    pad_token_id: int = schema_utils.Integer(
        default=1,
        description="The ID of the token to use as padding.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["pad_token_id"],
    )

    gradient_checkpointing: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use gradient checkpointing.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["gradient_checkpointing"],
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="Type of position embedding.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["position_embedding_type"],
    )

    classifier_dropout: float = schema_utils.FloatRange(
        default=None,
        allow_none=True,
        min=0,
        max=1,
        description="The dropout ratio for the classification head.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["classifier_dropout"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["CamemBERT"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("t5", TEXT)
@ludwig_dataclass
class T5Config(HFEncoderConfig):
    """This dataclass configures the schema used for an T5 encoder."""

    @staticmethod
    def module_name():
        return "T5"

    type: str = schema_utils.ProtectedString(
        "t5",
        description=ENCODER_METADATA["T5"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["T5"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["T5"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="t5-small",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["T5"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["T5"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["T5"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["T5"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["T5"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=32128,
        description="Vocabulary size of the T5 model. Defines the number of different tokens that can be represented "
        "by the inputs_ids passed when calling T5Model or TFT5Model.",
        parameter_metadata=ENCODER_METADATA["T5"]["vocab_size"],
    )

    d_model: int = schema_utils.PositiveInteger(
        default=512,
        description="Size of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["T5"]["d_model"],
    )

    d_kv: int = schema_utils.PositiveInteger(
        default=64,
        description="Size of the key, query, value projections per attention head. d_kv has to be equal to d_model // "
        "num_heads.",
        parameter_metadata=ENCODER_METADATA["T5"]["d_kv"],
    )

    d_ff: int = schema_utils.PositiveInteger(
        default=2048,
        description="Size of the intermediate feed forward layer in each T5Block.",
        parameter_metadata=ENCODER_METADATA["T5"]["d_ff"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["T5"]["num_layers"],
    )

    num_decoder_layers: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of hidden layers in the Transformer decoder. Will use the same value as num_layers if not "
        "set.",
        parameter_metadata=ENCODER_METADATA["T5"]["num_decoder_layers"],
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["T5"]["num_heads"],
    )

    relative_attention_num_buckets: int = schema_utils.PositiveInteger(
        default=32,
        description="The number of buckets to use for each attention layer.",
        parameter_metadata=ENCODER_METADATA["T5"]["relative_attention_num_buckets"],
    )

    dropout_rate: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The ratio for all dropout layers.",
        parameter_metadata=ENCODER_METADATA["T5"]["dropout_rate"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-6,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["T5"]["layer_norm_eps"],
    )

    initializer_factor: float = schema_utils.NonNegativeFloat(
        default=1,
        description="A factor for initializing all weight matrices (should be kept to 1, used internally for "
        "initialization testing).",
        parameter_metadata=ENCODER_METADATA["T5"]["initializer_factor"],
    )

    feed_forward_proj: str = schema_utils.StringOptions(
        ["relu", "gated-gelu"],
        default="relu",
        description="Type of feed forward layer to be used. Should be one of 'relu' or 'gated-gelu'. T5v1.1 uses the "
        "'gated-gelu' feed forward projection. Original T5 uses 'relu'.",
        parameter_metadata=ENCODER_METADATA["T5"]["feed_forward_proj"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["T5"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("flaubert", TEXT)
@ludwig_dataclass
class FlauBERTConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an FlauBERT encoder."""

    @staticmethod
    def module_name():
        return "FlauBERT"

    type: str = schema_utils.ProtectedString(
        "flaubert",
        description=ENCODER_METADATA["FlauBERT"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="flaubert/flaubert_small_cased",
        description="Name of path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30145,
        description="Vocabulary size of the FlauBERT model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling FlaubertModel or TFFlaubertModel.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["vocab_size"],
    )

    pre_norm: bool = schema_utils.Boolean(
        default=True,
        description="Whether to apply the layer normalization before or after the feed forward layer following the "
        "attention in each layer (Vaswani et al., Tensor2Tensor for Neural Machine Translation. 2018)",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["pre_norm"],
    )

    layerdrop: float = schema_utils.FloatRange(
        default=0.2,
        min=0,
        max=1,
        description="Probability to drop layers during training (Fan et al., Reducing Transformer Depth on Demand "
        "with Structured Dropout. ICLR 2020)",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["layerdrop"],
    )

    emb_dim: int = schema_utils.PositiveInteger(
        default=512,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["emb_dim"],
    )

    n_layers: int = schema_utils.PositiveInteger(
        default=6,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["n_layers"],
    )

    n_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["n_heads"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["dropout"],
    )

    attention_dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for the attention mechanism",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["attention_dropout"],
    )

    gelu_activation: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not to use a gelu activation instead of relu.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["gelu_activation"],
    )

    sinusoidal_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["sinusoidal_embeddings"],
    )

    causal: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not the model should behave in a causal manner. Causal models use a triangular "
        "attention mask in order to only attend to the left-side context instead if a bidirectional "
        "context.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["causal"],
    )

    asm: bool = schema_utils.Boolean(
        default=False,
        description="Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the "
        "prediction layer.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["asm"],
    )

    n_langs: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of languages the model handles. Set to 1 for monolingual models.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["n_langs"],
    )

    use_lang_emb: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use language embeddings. Some models use additional language embeddings, "
        "see the multilingual models page for information on how to use them.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["use_lang_emb"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["max_position_embeddings"],
    )

    embed_init_std: float = schema_utils.NonNegativeFloat(
        default=2048**-0.5,
        description="The standard deviation of the truncated_normal_initializer for initializing the embedding "
        "matrices.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["embed_init_std"],
    )

    init_std: int = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices "
        "except the embedding matrices.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["init_std"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-06,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["layer_norm_eps"],
    )

    bos_index: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The index of the beginning of sentence token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["bos_index"],
    )

    eos_index: int = schema_utils.NonNegativeInteger(
        default=1,
        description="The index of the end of sentence token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["eos_index"],
    )

    pad_index: int = schema_utils.NonNegativeInteger(
        default=2,
        description="The index of the padding token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["pad_index"],
    )

    unk_index: int = schema_utils.NonNegativeInteger(
        default=3,
        description="The index of the unknown token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["unk_index"],
    )

    mask_index: int = schema_utils.NonNegativeInteger(
        default=5,
        description="The index of the masking token in the vocabulary.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["mask_index"],
    )

    is_encoder: bool = schema_utils.Boolean(
        default=True,
        description="Whether or not the initialized model should be a transformer encoder or decoder as seen in "
        "Vaswani et al.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["is_encoder"],
    )

    mask_token_id: int = schema_utils.Integer(
        default=0,
        description="Model agnostic parameter to identify masked tokens when generating text in an MLM context.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["mask_token_id"],
    )

    lang_id: int = schema_utils.Integer(
        default=0,
        description="The ID of the language used by the model. This parameter is used when generating text in a given "
        "language.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["lang_id"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["FlauBERT"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("electra", TEXT)
@ludwig_dataclass
class ELECTRAConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an ELECTRA encoder."""

    @staticmethod
    def module_name():
        return "ELECTRA"

    type: str = schema_utils.ProtectedString(
        "electra",
        description=ENCODER_METADATA["ELECTRA"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["use_pretrained"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="google/electra-small-discriminator",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["pretrained_model_name_or_path"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Are the pretrained encoder weights saved in this model's checkpoint? Automatically set to"
        "True for trained models to prevent loading pretrained encoder weights from model hub.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["saved_weights_in_checkpoint"],
    )

    reduce_output: str = schema_utils.String(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=30522,
        description="Vocabulary size of the ELECTRA model. Defines the number of different tokens that can be "
        "represented by the inputs_ids passed when calling ElectraModel or TFElectraModel.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["vocab_size"],
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["embedding_size"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Dimensionality of the encoder layers and the pooler layer.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["hidden_size"],
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=12,
        description="Number of hidden layers in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["num_hidden_layers"],
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=4,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["num_attention_heads"],
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=1024,
        description="Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["intermediate_size"],
    )

    hidden_act: Union[str, Callable] = schema_utils.StringOptions(  # TODO: add support for callable
        ["gelu", "relu", "silu", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["hidden_act"],
    )

    hidden_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["hidden_dropout_prob"],
    )

    attention_probs_dropout_prob: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="The dropout ratio for the attention probabilities.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["attention_probs_dropout_prob"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=2,
        description="The vocabulary size of the token_type_ids passed when calling ElectraModel or TFElectraModel.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["type_vocab_size"],
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["initializer_range"],
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["layer_norm_eps"],
    )

    position_embedding_type: str = schema_utils.StringOptions(
        ["absolute", "relative_key", "relative_key_query"],
        default="absolute",
        description="Type of position embedding.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["position_embedding_type"],
    )

    classifier_dropout: float = schema_utils.FloatRange(
        default=None,
        allow_none=True,
        min=0,
        max=1,
        description="The dropout ratio for the classification head.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["classifier_dropout"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["ELECTRA"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("longformer", TEXT)
@ludwig_dataclass
class LongformerConfig(HFEncoderConfig):
    """This dataclass configures the schema used for a Longformer encoder."""

    @staticmethod
    def module_name():
        return "Longformer"

    type: str = schema_utils.ProtectedString(
        "longformer",
        description=ENCODER_METADATA["Longformer"]["type"].long_description,
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["Longformer"]["max_sequence_length"],
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use the pretrained weights for the model. If false, the model will train from "
        "scratch which is very computationally expensive.",
        parameter_metadata=ENCODER_METADATA["Longformer"]["use_pretrained"],
    )

    attention_window: Union[List[int], int] = schema_utils.OneOfOptionsField(
        default=512,
        allow_none=True,
        description="Size of an attention window around each token. If an int, use the same size for all layers. To "
        "specify a different window size for each layer, use a List[int] where len(attention_window) == "
        "num_hidden_layers.",
        field_options=[
            schema_utils.PositiveInteger(allow_none=True, description="", default=None),
            schema_utils.List(list_type=int, allow_none=False),
        ],
        parameter_metadata=ENCODER_METADATA["Longformer"]["attention_window"],
    )

    sep_token_id: int = schema_utils.Integer(
        default=2,
        description="ID of the separator token, which is used when building a sequence from multiple sequences",
        parameter_metadata=ENCODER_METADATA["Longformer"]["sep_token_id"],
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="allenai/longformer-base-4096",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["Longformer"]["pretrained_model_name_or_path"],
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
        parameter_metadata=ENCODER_METADATA["Longformer"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["Longformer"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["Longformer"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=50265,
        description="Vocabulary size of the Longformer model.",
        parameter_metadata=ENCODER_METADATA["Longformer"]["vocab_size"],
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=4098,
        description="The maximum sequence length that this model might ever be used with. Typically set this to "
        "something large just in case (e.g., 512 or 1024 or 2048).",
        parameter_metadata=ENCODER_METADATA["Longformer"]["max_position_embeddings"],
    )

    type_vocab_size: int = schema_utils.PositiveInteger(
        default=1,
        description="The vocabulary size of the token_type_ids passed when calling LongformerEncoder",
        parameter_metadata=ENCODER_METADATA["Longformer"]["type_vocab_size"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["Longformer"]["pretrained_kwargs"],
    )


@DeveloperAPI
@register_encoder_config("auto_transformer", TEXT)
@ludwig_dataclass
class AutoTransformerConfig(HFEncoderConfig):
    """This dataclass configures the schema used for an AutoTransformer encoder."""

    @staticmethod
    def module_name():
        return "AutoTransformer"

    type: str = schema_utils.ProtectedString(
        "auto_transformer",
        description=ENCODER_METADATA["AutoTransformer"]["type"].long_description,
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="bert-base-uncased",
        description="Name or path of the pretrained model.",
        parameter_metadata=ENCODER_METADATA["AutoTransformer"]["pretrained_model_name_or_path"],
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Maximum length of the input sequence.",
        parameter_metadata=ENCODER_METADATA["AutoTransformer"]["max_sequence_length"],
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="sum",
        description="The method used to reduce a sequence of tensors down to a single tensor.",
        parameter_metadata=ENCODER_METADATA["AutoTransformer"]["reduce_output"],
    )

    trainable: bool = schema_utils.Boolean(
        default=False,
        description="Whether to finetune the model on your dataset.",
        parameter_metadata=ENCODER_METADATA["AutoTransformer"]["trainable"],
    )

    vocab: list = schema_utils.List(
        default=None,
        description="Vocabulary for the encoder",
        parameter_metadata=ENCODER_METADATA["AutoTransformer"]["vocab"],
    )

    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description=(
            "Vocabulary size of the AutoTransformer model. If None, the vocab size will be inferred "
            "from the given pretrained model"
        ),
        parameter_metadata=ENCODER_METADATA["AutoTransformer"]["vocab_size"],
    )

    pretrained_kwargs: dict = schema_utils.Dict(
        default=None,
        description="Additional kwargs to pass to the pretrained model.",
        parameter_metadata=ENCODER_METADATA["AutoTransformer"]["pretrained_kwargs"],
    )
