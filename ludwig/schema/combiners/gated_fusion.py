from typing import Any

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import register_combiner_config


@DeveloperAPI
@register_combiner_config("gated_fusion")
class GatedFusionCombinerConfig(BaseCombinerConfig):
    """Parameters for gated fusion combiner."""

    type: str = schema_utils.ProtectedString(
        "gated_fusion",
        description="Combines features using a gating mechanism that learns to weight each input feature's "
        "contribution through sigmoid gates.",
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Hidden size of the gating layers. Each input feature is projected to this size.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="Dropout rate for the gating layers.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of the fully connected layer after gated fusion.",
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
