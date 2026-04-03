from typing import Any

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import register_combiner_config


@DeveloperAPI
@register_combiner_config("cross_attention")
class CrossAttentionCombinerConfig(BaseCombinerConfig):
    """Parameters for cross-attention combiner."""

    type: str = schema_utils.ProtectedString(
        "cross_attention",
        description="Combines features using cross-attention, where each feature attends to all other features "
        "through a multi-head cross-attention mechanism.",
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Hidden size of the cross-attention layers. Each input feature is projected to this size.",
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads in the cross-attention layers.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="Number of stacked cross-attention layers.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="Dropout rate for the cross-attention layers.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of the fully connected layer after cross-attention.",
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
