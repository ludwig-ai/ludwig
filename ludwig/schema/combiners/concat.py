from typing import Any, Dict, List, Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class ConcatCombinerConfig(BaseCombinerConfig):
    """Parameters for concat combiner."""

    @staticmethod
    def module_name():
        return "ConcatCombiner"

    type: str = schema_utils.ProtectedString(
        "concat",
        description=COMBINER_METADATA["ConcatCombiner"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField(parameter_metadata=COMBINER_METADATA["ConcatCombiner"]["dropout"])

    activation: str = schema_utils.ActivationOptions(default="relu")

    flatten_inputs: bool = schema_utils.Boolean(
        default=False,
        description="Whether to flatten input tensors to a vector.",
        parameter_metadata=COMBINER_METADATA["ConcatCombiner"]["flatten_inputs"],
    )

    residual: bool = schema_utils.Boolean(
        default=False,
        description=(
            "Whether to add a residual connection to each fully connected layer block. "
            "Requires all fully connected layers to have the same `output_size`."
        ),
        parameter_metadata=COMBINER_METADATA["ConcatCombiner"]["residual"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=COMBINER_METADATA["ConcatCombiner"]["use_bias"],
    )

    bias_initializer: Union[str, Dict] = common_fields.BiasInitializerField()

    weights_initializer: Union[str, Dict] = common_fields.WeightsInitializerField()

    num_fc_layers: int = common_fields.NumFCLayersField()

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of a fully connected layer.",
        parameter_metadata=COMBINER_METADATA["ConcatCombiner"]["output_size"],
    )

    norm: Optional[str] = common_fields.NormField()

    norm_params: Optional[dict] = common_fields.NormParamsField()

    fc_layers: Optional[List[Dict[str, Any]]] = common_fields.FCLayersField()
