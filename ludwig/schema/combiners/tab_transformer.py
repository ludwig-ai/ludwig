from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_options import CommonTransformerConfig
from ludwig.schema.metadata.combiner_metadata import COMBINER_METADATA


@dataclass(repr=False, order=True)
class TabTransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for tab transformer combiner."""

    type: str = schema_utils.StringOptions(
        ["tabtransformer"],
        default="tabtransformer",
        allow_none=False,
        description="Type of combiner.",
    )

    embed_input_feature_name: Optional[Union[str, int]] = schema_utils.Embed(
        description="This value controls the size of the embeddings. Valid values are `add` which uses the "
        "`hidden_size` value or an integer that is set to a specific value. In the case of an integer "
        "value, it must be smaller than hidden_size.",
        parameter_metadata=COMBINER_METADATA["TabTransformerCombiner"]["embed_input_feature_name"],
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="concat",
        description="",
        parameter_metadata=COMBINER_METADATA["TabTransformerCombiner"]["reduce_output"],
    )
