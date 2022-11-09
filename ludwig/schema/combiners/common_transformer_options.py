from typing import Any, Dict, List, Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.combiner_metadata import COMBINER_METADATA


@dataclass(repr=False, order=True)
class CommonTransformerConfig:
    """Common transformer parameter values."""

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="Dropout rate for the transformer block.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["dropout"],
    )

    transformer_output_size: int = schema_utils.NonNegativeInteger(
        default=256,
        description="Size of the fully connected layer after self attention in the transformer block. This is usually "
        "the same as `hidden_size` and `embedding_size`.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["transformer_output_size"],
    )

    hidden_size: int = schema_utils.NonNegativeInteger(
        default=256,
        description="The number of hidden units of the TransformerStack as well as the dimension that each incoming "
        "input feature is projected to before feeding to the TransformerStack.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["hidden_size"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of transformer layers",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["num_layers"],
    )

    num_heads: int = schema_utils.NonNegativeInteger(
        default=8,
        description="Number of heads of the self attention in the transformer block.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["num_heads"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["use_bias"],
    )

    bias_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="zeros",
        description="",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["bias_initializer"],
    )

    weights_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="xavier_uniform",
        description="",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["weights_initializer"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of a fully connected layer.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["output_size"],
    )

    norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        description="",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["norm"],
    )

    norm_params: Optional[dict] = schema_utils.Dict(
        description="",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["norm_params"],
    )

    # TODO(#1673): Add conditional logic for fields like this one:
    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of stacked fully connected layers (only applies if `reduce_output` is not null).",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["num_fc_layers"],
    )

    fc_layers: Optional[List[Dict[str, Any]]] = schema_utils.DictList(
        description="",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["fc_layers"],
    )

    fc_dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["fc_dropout"],
    )

    fc_activation: str = schema_utils.ActivationOptions(
        default="relu",
        description="",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["fc_activation"],
    )

    fc_residual: bool = schema_utils.Boolean(
        default=False,
        description="",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["fc_residual"],
    )
