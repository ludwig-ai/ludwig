from typing import Any, Dict, List, Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig


@dataclass
class ComparatorCombinerConfig(BaseCombinerConfig):
    """Parameters for comparator combiner."""

    entity_1: List[str]
    """TODO: Document parameters."""

    entity_2: List[str]
    """TODO: Document parameters."""

    fc_layers: Optional[List[Dict[str, Any]]] = schema_utils.DictList(description="")

    num_fc_layers: int = schema_utils.NonNegativeInteger(default=1, description="")

    output_size: int = schema_utils.PositiveInteger(default=256, description="Output size of a fully connected layer")

    use_bias: bool = schema_utils.Boolean(default=True, description="Whether the layer uses a bias vector.")

    weights_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(default="xavier_uniform", description="")

    bias_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(default="zeros", description="")

    norm: Optional[str] = schema_utils.StringOptions(["batch", "layer"], description="")

    norm_params: Optional[dict] = schema_utils.Dict(description="")

    activation: str = schema_utils.ActivationOptions(default="relu", description="")

    dropout: float = schema_utils.FloatRange(
        default=0.0, min=0, max=1, description="Dropout rate for the transformer block."
    )
