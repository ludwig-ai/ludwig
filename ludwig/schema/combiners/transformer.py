from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_options import CommonTransformerConfig
from ludwig.schema.metadata import COMBINER_METADATA


@DeveloperAPI
@dataclass(repr=False, order=True)
class TransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for transformer combiner."""

    type: str = schema_utils.ProtectedString(
        "transformer",
        description="The transformer combiner combines input features using a stack of Transformer blocks (from "
        "Attention Is All You Need). It assumes all outputs from encoders are tensors of size `b x h` "
        "where `b` is the batch size and `h` is the hidden dimension, which can be different for each "
        "input. If the input tensors have a different shape, it automatically flattens them. It then "
        "projects each input tensor to the same hidden / embedding size and encodes them with a stack of "
        "Transformer layers. Finally, the transformer combiner applies a reduction to the outputs of the "
        "Transformer stack, followed by optional fully connected layers. The output is a `b x h` tensor "
        "where `h` is the size of the last fully connected layer or the hidden / embedding size, or a "
        "`b x n x h` where `n` is the number of input features and `h` is the hidden / embedding size if "
        "no reduction is applied.",
    )

    reduce_output: Optional[str] = schema_utils.ReductionOptions(
        default="mean", description="", parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["reduce_output"]
    )
