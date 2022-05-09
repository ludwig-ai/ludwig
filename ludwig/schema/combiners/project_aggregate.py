from typing import Any, Dict, List, Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig


@dataclass
class ProjectAggregateCombinerConfig(BaseCombinerConfig):
    projection_size: int = schema_utils.PositiveInteger(
        default=128, description="All combiner inputs are projected to this size before being aggregated."
    )
    fc_layers: Optional[List[Dict[str, Any]]] = schema_utils.DictList(
        description="Full secification of the fully connected layers after the aggregation. "
        "It should be a list of dict, each disct representing one layer."
    )
    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=2, description="Number of fully connected layers after aggregation."
    )
    output_size: int = schema_utils.PositiveInteger(
        default=128, description="Output size of each layer of the stack of fully connected layers."
    )
    use_bias: bool = schema_utils.Boolean(default=True, description="Whether the layers use a bias vector.")
    weights_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="xavier_uniform",
        description="Initializer to use for the weights of the projection and for the fully connected layers.",
    )
    bias_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="zeros",
        description="Initializer to use for the baias of the projection and for the fully connected layers.",
    )
    norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default="layer",
        description="Normalization to apply to each projection and fully connected layer.",
    )
    norm_params: Optional[dict] = schema_utils.Dict(
        description="Parameters of the normalization to apply to each projection and fully connected layer."
    )
    activation: str = schema_utils.ActivationOptions(
        default="relu", description="Activation to apply to each fully connected layer."
    )
    dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="Dropout rate to apply to each fully connected layer."
    )
    residual: bool = schema_utils.Boolean(
        default=True,
        description="Whether to add residual skip connection between the fully connected layers in the stack..",
    )
