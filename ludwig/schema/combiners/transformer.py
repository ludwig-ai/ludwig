from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_options import CommonTransformerConfig


@dataclass
class TransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for transformer combiner."""

    reduce_output: Optional[str] = schema_utils.ReductionOptions(default="mean", description="")
