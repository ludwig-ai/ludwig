from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.common_transformer_options import CommonTransformerConfig
from ludwig.schema.metadata import COMBINER_METADATA


@DeveloperAPI
@dataclass(repr=False, order=True)
class TabTransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for tab transformer combiner."""

    type: str = schema_utils.ProtectedString(
        "tabtransformer",
        description="The tabtransformer combiner combines input features in the following sequence of operations. "
        "Except for binary and number features, the combiner projects features to an embedding size. "
        "These features are concatenated as if they were a sequence and passed through a transformer. "
        "After the transformer, the number and binary features are concatenated (which are of size 1) and "
        "then concatenated with the output of the transformer and is passed to a stack of fully connected "
        "layers (from TabTransformer: Tabular Data Modeling Using Contextual Embeddings). It assumes all "
        "outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the "
        "hidden dimension, which can be different for each input. If the input tensors have a different "
        "shape, it automatically flattens them. It then projects each input tensor to the same hidden / "
        "embedding size and encodes them with a stack of Transformer layers. Finally, the transformer "
        "combiner applies a reduction to the outputs of the Transformer stack, followed by the above "
        "concatenation and optional fully connected layers. The output is a `b x h` tensor where `h` is the"
        " size of the last fully connected layer or the hidden / embedding size, or a `b x n x h` where `n`"
        " is the number of input features and `h` is the hidden / embedding size if no reduction is "
        "applied.",
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
