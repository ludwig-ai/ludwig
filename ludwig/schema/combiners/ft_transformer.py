from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_options import CommonTransformerConfig
from ludwig.schema.combiners.utils import register_combiner_config


@DeveloperAPI
@register_combiner_config("ft_transformer")
class FTTransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """FT-Transformer combiner: project each feature to a token, prepend [CLS], apply Transformer self-attention.

    Output is the [CLS] token embedding, optionally followed by FC layers. Based on Gorishniy et al., NeurIPS 2021.
    """

    type: str = schema_utils.ProtectedString(
        "ft_transformer",
        description="FT-Transformer combiner with [CLS] token aggregation.",
    )
