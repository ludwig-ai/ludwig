from typing import Any, Dict, List, Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils
from ludwig.schema.combiners.base_combiner_config import BaseCombinerConfig


@dataclass
class ConcatCombinerConfig(BaseCombinerConfig):
    """Parameters for concat combiner."""

    fc_layers: Optional[List[Dict[str, Any]]] = utils.DictList(description="TODO: Document parameters.")

    num_fc_layers: int = utils.NonNegativeInteger(default=0, description="TODO: Document parameters.")

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

    activation: str = utils.ActivationOptions(default="relu", description="TODO: Document parameters.")

    dropout: float = utils.FloatRange(default=0.0, min=0, max=1, description="TODO: Document parameters.")

    flatten_inputs: bool = utils.Boolean(default=False, description="Whether to flatten input tensors to a vector.")

    residual: bool = utils.Boolean(
        default=False,
        description=(
            "Whether to add a residual connection to each fully connected layer block. All fully connected layers must"
            " have the same size"
        ),
    )
