from typing import Any, Dict, List, Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils


@dataclass
class CommonTransformerConfig:
    """Common transformer parameter values."""

    num_layers: int = schema_utils.PositiveInteger(default=1, description="")

    hidden_size: int = schema_utils.NonNegativeInteger(
        default=256,
        description=(
            "The number of hidden units of the TransformerStack as well as the dimension that each incoming input "
            "feature is projected to before feeding to the TransformerStack"
        ),
    )

    num_heads: int = schema_utils.NonNegativeInteger(
        default=8, description="Number of heads of the self attention in the transformer block."
    )

    transformer_output_size: int = schema_utils.NonNegativeInteger(
        default=256,
        description=(
            "Size of the fully connected layer after self attention in the transformer block. This is usually the same "
            "as `hidden_size` and `embedding_size`."
        ),
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1, min=0, max=1, description="Dropout rate for the transformer block."
    )

    fc_layers: Optional[List[Dict[str, Any]]] = schema_utils.DictList(description="")

    # TODO(#1673): Add conditional logic for fields like this one:
    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of stacked fully connected layers (only applies if `reduce_output` is not null).",
    )

    output_size: int = schema_utils.PositiveInteger(default=256, description="Output size of a fully connected layer.")

    use_bias: bool = schema_utils.Boolean(default=True, description="Whether the layer uses a bias vector.")

    weights_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(default="xavier_uniform", description="")

    bias_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(default="zeros", description="")

    norm: Optional[str] = schema_utils.StringOptions(["batch", "layer"], description="")

    norm_params: Optional[dict] = schema_utils.Dict(description="")

    fc_activation: str = schema_utils.ActivationOptions(default="relu", description="")

    fc_dropout: float = schema_utils.FloatRange(default=0.0, min=0, max=1, description="")

    fc_residual: bool = schema_utils.Boolean(default=False, description="")
