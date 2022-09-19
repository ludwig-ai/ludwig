from typing import Any, Dict, List, Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.metadata.combiner_metadata import COMBINER_METADATA


@dataclass
class ComparatorCombinerConfig(BaseCombinerConfig):
    """Parameters for comparator combiner."""

    entity_1: List[str]
    """TODO: Document parameters."""

    entity_2: List[str]
    """TODO: Document parameters."""

    fc_layers: Optional[List[Dict[str, Any]]] = schema_utils.DictList(
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["fc_layers"],
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=1,
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["num_fc_layers"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of a fully connected layer",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["output_size"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["use_bias"],
    )

    weights_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="xavier_uniform",
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["weights_initializer"],
    )

    bias_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="zeros",
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["bias_initializer"],
    )

    norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["norm"],
    )

    norm_params: Optional[dict] = schema_utils.Dict(
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["norm_params"],
    )

    activation: str = schema_utils.ActivationOptions(
        default="relu",
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["activation"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate for the transformer block.",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["dropout"],
    )
