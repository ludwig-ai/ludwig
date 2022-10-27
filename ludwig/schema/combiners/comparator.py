from typing import Any, Dict, List, Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.metadata.combiner_metadata import COMBINER_METADATA


@dataclass(repr=False, order=True)
class ComparatorCombinerConfig(BaseCombinerConfig):
    """Parameters for comparator combiner."""

    type: str = schema_utils.StringOptions(
        ["comparator"],
        default="comparator",
        allow_none=False,
        description="Type of combiner.",
    )

    entity_1: List[str] = schema_utils.List(
        default=None,
        description="The list of input features composing the first entity to compare.",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["entity_1"],
    )

    entity_2: List[str] = schema_utils.List(
        default=None,
        description="The list of input features composing the second entity to compare.",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["entity_2"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate for the transformer block.",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["dropout"],
    )

    activation: str = schema_utils.ActivationOptions(
        default="relu",
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["activation"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["use_bias"],
    )

    bias_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="zeros",
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["bias_initializer"],
    )

    weights_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="xavier_uniform",
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["weights_initializer"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of a fully connected layer",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["output_size"],
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

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=1,
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["num_fc_layers"],
    )

    fc_layers: Optional[List[Dict[str, Any]]] = schema_utils.DictList(
        description="",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["fc_layers"],
    )
