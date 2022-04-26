from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils
from ludwig.schema.combiners.base_combiner_config import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_config import CommonTransformerConfig


@dataclass
class TransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for transformer combiner."""

    reduce_output: Optional[str] = utils.ReductionOptions(default="mean", description="TODO: Document parameters.")
