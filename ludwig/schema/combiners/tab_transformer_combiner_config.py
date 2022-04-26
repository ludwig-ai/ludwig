from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils
from ludwig.schema.combiners.base_combiner_config import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_config import CommonTransformerConfig


@dataclass
class TabTransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for tab transformer combiner."""

    embed_input_feature_name: Optional[Union[str, int]] = utils.Embed()

    reduce_output: str = utils.ReductionOptions(default="concat", description="TODO: Document parameters.")
