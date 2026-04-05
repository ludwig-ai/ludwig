
from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import register_combiner_config


@DeveloperAPI
@register_combiner_config("hypernetwork")
class HyperNetworkCombinerConfig(BaseCombinerConfig):
    """HyperNetwork combiner: one modality generates weights for processing another.

    Based on HyperFusion (arXiv 2403.13319, 2024).
    """

    type: str = schema_utils.ProtectedString(
        "hypernetwork",
        description="HyperNetwork combiner where one feature generates processing weights for others.",
    )

    hidden_size: int = schema_utils.PositiveInteger(default=128, description="Hidden size for feature projections.")

    hyper_hidden_size: int = schema_utils.PositiveInteger(
        default=64, description="Hidden size inside the hypernetwork weight generator."
    )

    output_size: int = schema_utils.PositiveInteger(default=128, description="Output size of the FC stack.")

    num_fc_layers: int = common_fields.NumFCLayersField()

    dropout: float = schema_utils.FloatRange(default=0.1, min=0, max=1, description="Dropout rate.")

    activation: str = schema_utils.ActivationOptions(default="relu")
