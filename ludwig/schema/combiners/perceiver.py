from typing import Any

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import register_combiner_config


@DeveloperAPI
@register_combiner_config("perceiver")
class PerceiverCombinerConfig(BaseCombinerConfig):
    """Parameters for Perceiver combiner."""

    type: str = schema_utils.ProtectedString(
        "perceiver",
        description="Combines features using a Perceiver-style architecture with learned latent queries that "
        "cross-attend to the input features, followed by self-attention layers.",
    )

    num_latents: int = schema_utils.PositiveInteger(
        default=32,
        description="Number of learned latent query vectors.",
    )

    latent_dim: int = schema_utils.PositiveInteger(
        default=256,
        description="Dimensionality of each latent query vector.",
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads in the cross-attention and self-attention layers.",
    )

    num_self_attention_layers: int = schema_utils.PositiveInteger(
        default=2,
        description="Number of self-attention layers applied to the latent queries after cross-attention.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="Dropout rate for the attention layers.",
    )

    reduce_output: str | None = schema_utils.ReductionOptions(
        default="mean",
        description="Strategy to use to aggregate the latent vectors before the FC stack.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of the fully connected layer after the Perceiver block.",
    )

    num_fc_layers: int = common_fields.NumFCLayersField()

    fc_layers: list[dict[str, Any]] | None = common_fields.FCLayersField()

    activation: str = schema_utils.ActivationOptions(default="relu")

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
    )

    weights_initializer: str | dict = common_fields.WeightsInitializerField()

    bias_initializer: str | dict = common_fields.BiasInitializerField()

    norm: str | None = common_fields.NormField()

    norm_params: dict | None = common_fields.NormParamsField()
