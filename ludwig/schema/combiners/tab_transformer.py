from typing import Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_options import CommonTransformerConfig
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class TabTransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for tab transformer combiner."""

    @staticmethod
    def module_name():
        return "TabTransformerCombiner"

    type: str = schema_utils.ProtectedString(
        "tabtransformer",
        description=COMBINER_METADATA["TabTransformerCombiner"]["type"].long_description,
    )

    embed_input_feature_name: Optional[Union[str, int]] = schema_utils.Embed(
        description="This value controls the size of the embeddings. Valid values are `add` which uses the "
        "`hidden_size` value or an integer that is set to a specific value. In the case of an integer "
        "value, it must be smaller than hidden_size.",
        parameter_metadata=COMBINER_METADATA["TabTransformerCombiner"]["embed_input_feature_name"],
    )

    reduce_output: str = schema_utils.ReductionOptions(
        default="concat",
        description="Strategy to use to aggregate the output of the transformer.",
        parameter_metadata=COMBINER_METADATA["TabTransformerCombiner"]["reduce_output"],
    )
