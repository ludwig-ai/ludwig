from typing import List, Set

from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
from ludwig.schema.utils import ludwig_dataclass

"""
NOTE TO DEVELOPERS: the implementation of the schema classes below must match the parameters of the HF PretrainedConfig
class exactly. This is because we convert this object into the matching HF PretrainedConfig object before passing it to
the model. Additionally, for loading and saving pretrained models, we take the config from the existing model and load
it into this config before saving. As such, if any params needed by the pretrained model are missing, we will not be
able to load checkpoints correctly.

A common mistake is to look at the PretrainedConfig __init__ method params and ignore any additional **kwargs. In some
cases, these kwargs are used to set additional params on the config object. For example, the DebertaConfig class has
`position_buckets` as a kwarg param, but it nonetheless requires this to construct the model architecture.

To debug issues with missing parameters, try printing out the `model.config` of the pretrained transformer and check
for any params it includes that are not present in your schema config.
"""


@ludwig_dataclass
class DebertaModelParams(schema_utils.BaseMarshmallowConfig):
    @classmethod
    def get_hf_config_param_names(cls) -> Set[str]:
        return DebertaModelParams.get_valid_field_names()

    # Model architecture params for training from scratch
    # TODO(travis): conditionally disable setting these when `use_pretrained=True`.
    vocab_size: int = schema_utils.PositiveInteger(
        default=None,
        description="",
        parameter_metadata=INTERNAL_ONLY,
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=1536,
        description="Dimensionality of the encoder layers and the pooler layer.",
    )

    num_hidden_layers: int = schema_utils.PositiveInteger(
        default=24,
        description="Number of hidden layers in the Transformer encoder.",
    )

    num_attention_heads: int = schema_utils.PositiveInteger(
        default=24,
        description="Number of attention heads for each attention layer in the Transformer encoder.",
    )

    intermediate_size: int = schema_utils.PositiveInteger(
        default=6144,
        description="Dimensionality of the 'intermediate' (often named feed-forward) layer in the Transformer encoder.",
    )

    hidden_act: str = schema_utils.StringOptions(
        options=["gelu", "relu", "silu", "tanh", "gelu_fast", "mish", "linear", "sigmoid", "gelu_new"],
        default="gelu",
        description="The non-linear activation function (function or string) in the encoder and pooler.",
    )

    hidden_dropout_prob: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.",
    )

    attention_probs_dropout_prob: float = schema_utils.NonNegativeFloat(
        default=0.1,
        description="The dropout ratio for the attention probabilities.",
    )

    max_position_embeddings: int = schema_utils.PositiveInteger(
        default=512,
        description=(
            "The maximum sequence length that this model might ever be used with. Typically set this to something "
            "large just in case (e.g., 512 or 1024 or 2048)."
        ),
    )

    type_vocab_size: int = schema_utils.NonNegativeInteger(
        default=0,
        description=("The vocabulary size of the `token_type_ids`."),
    )

    initializer_range: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description=(
            "The standard deviation of the truncated_normal_initializer for initializing all weight matrices."
        ),
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-7,
        description="The epsilon used by the layer normalization layers.",
    )

    relative_attention: bool = schema_utils.Boolean(
        default=True,
        description="Whether use relative position encoding.",
    )

    max_relative_positions: int = schema_utils.Integer(
        default=-1,
        description=(
            "The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same "
            "value as `max_position_embeddings`."
        ),
    )

    pad_token_id: int = schema_utils.Integer(
        default=0,
        description="The value used to pad input_ids.",
    )

    position_biased_input: bool = schema_utils.Boolean(
        default=False,
        description="Whether add absolute position embedding to content embedding.",
    )

    pos_att_type: List[str] = schema_utils.List(
        default=["p2c", "c2p"],
        description=(
            "The type of relative position attention, it can be a combination of `['p2c', 'c2p']`, e.g. `['p2c']`, "
            "`['p2c', 'c2p']`, `['p2c', 'c2p']`."
        ),
    )

    layer_norm_eps: float = schema_utils.NonNegativeFloat(
        default=1e-12,
        description="The epsilon used by the layer normalization layers.",
    )

    pooler_hidden_size: int = schema_utils.PositiveInteger(
        default=1536,
        description="The hidden size of the pooler layers.",
    )

    pooler_dropout: float = schema_utils.NonNegativeFloat(
        default=0,
        description="The dropout ratio for the pooler layers.",
    )

    pooler_hidden_act: str = schema_utils.StringOptions(
        options=["gelu", "relu", "silu", "tanh", "gelu_fast", "mish", "linear", "sigmoid", "gelu_new"],
        default="gelu",
        description="The activation function (function or string) in the pooler.",
    )

    position_buckets: int = schema_utils.PositiveInteger(
        default=256,
        description="The number of buckets to use for each attention layer.",
    )

    share_att_key: bool = schema_utils.Boolean(
        default=True,
        description="Whether to share attention key across layers.",
    )

    norm_rel_ebd: str = schema_utils.StringOptions(
        options=["layer_norm", "none"],
        default="layer_norm",
        description="The normalization method for relative embeddings.",
    )
