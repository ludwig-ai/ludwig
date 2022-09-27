from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_options import CommonTransformerConfig
from ludwig.schema.metadata.combiner_metadata import COMBINER_METADATA


@dataclass
class TabTransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for tab transformer combiner."""

    embed_input_feature_name: Optional[Union[str, int]] = schema_utils.Embed(
        description="Must be an integer, 'add', or null. If an integer, specifies the embedding size for input "
        "feature names. Input feature name embeddings will be concatenated to hidden representations. "
        "Must be less than or equal to hidden_size. If 'add', input feature names use embeddings the same "
        "size as hidden_size, and are added (element-wise) to the hidden representations. If null, "
        "input feature embeddings are not used.",
        parameter_metadata=COMBINER_METADATA["TabTransformerCombiner"]["embed_input_feature_name"],
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="concat",
        description="",
        parameter_metadata=COMBINER_METADATA["TabTransformerCombiner"]["reduce_output"],
    )
