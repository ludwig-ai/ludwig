from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_options import CommonTransformerConfig


@dataclass
class TabTransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for tab transformer combiner."""

    embed_input_feature_name: Optional[Union[str, int]] = schema_utils.Embed()

    reduce_output: str = schema_utils.ReductionOptions(default="concat", description="")
