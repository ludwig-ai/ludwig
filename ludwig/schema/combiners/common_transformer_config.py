from typing import Any, Dict, List, Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils


@dataclass
class CommonTransformerConfig:
    """Common transformer parameter values."""

    num_layers: int = utils.PositiveInteger(default=1, description="TODO: Document parameters.")

    hidden_size: int = utils.NonNegativeInteger(
        default=256,
        description=(
            "The number of hidden units of the TransformerStack as well as the dimension that each incoming input "
            "feature is projected to before feeding to the TransformerStack"
        ),
    )

    num_heads: int = utils.NonNegativeInteger(
        default=8, description="Number of heads of the self attention in the transformer block."
    )

    transformer_output_size: int = utils.NonNegativeInteger(
        default=256,
        description=(
            "Size of the fully connected layer after self attention in the transformer block. This is usually the same "
            "as `hidden_size` and `embedding_size`."
        ),
    )

    dropout: float = utils.FloatRange(default=0.1, min=0, max=1, description="Dropout rate for the transformer block.")

    fc_layers: Optional[List[Dict[str, Any]]] = utils.DictList(description="TODO: Document parameters.")

    # TODO(#1673): Add conditional logic for fields like this one:
    num_fc_layers: int = utils.NonNegativeInteger(
        default=0,
        description="The number of stacked fully connected layers (only applies if `reduce_output` is not null).",
    )

    output_size: int = utils.PositiveInteger(default=256, description="Output size of a fully connected layer.")

    use_bias: bool = utils.Boolean(default=True, description="Whether the layer uses a bias vector.")

    weights_initializer: Union[str, Dict] = utils.InitializerOrDict(
        default="xavier_uniform", description="TODO: Document parameters."
    )

    bias_initializer: Union[str, Dict] = utils.InitializerOrDict(
        default="zeros", description="TODO: Document parameters."
    )

    norm: Optional[str] = utils.StringOptions(["batch", "layer"], description="TODO: Document parameters.")

    norm_params: Optional[dict] = utils.Dict(description="TODO: Document parameters.")

    fc_activation: str = utils.ActivationOptions(default="relu", description="TODO: Document parameters.")

    fc_dropout: float = utils.FloatRange(default=0.0, min=0, max=1, description="TODO: Document parameters.")

    fc_residual: bool = utils.Boolean(default=False, description="TODO: Document parameters.")
