from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_options import CommonTransformerConfig
from ludwig.schema.combiners.utils import register_combiner_config
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_combiner_config("transformer")
@ludwig_dataclass
class TransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for transformer combiner."""

    type: str = schema_utils.ProtectedString(
        "transformer",
        description=COMBINER_METADATA["transformer"]["type"].long_description,
    )

    reduce_output: Optional[str] = schema_utils.ReductionOptions(
        default="mean",
        description="Strategy to use to aggregate the output of the transformer.",
        parameter_metadata=COMBINER_METADATA["transformer"]["reduce_output"],
    )
