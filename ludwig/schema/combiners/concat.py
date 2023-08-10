from typing import Any, Dict, List, Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import register_combiner_config
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_combiner_config("concat")
@ludwig_dataclass
class ConcatCombinerConfig(BaseCombinerConfig):
    """Parameters for concat combiner."""

    type: str = schema_utils.ProtectedString(
        "concat",
        description=COMBINER_METADATA["concat"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField()

    activation: str = schema_utils.ActivationOptions(default="relu")

    flatten_inputs: bool = schema_utils.Boolean(
        default=False,
        description="Whether to flatten input tensors to a vector.",
        parameter_metadata=COMBINER_METADATA["concat"]["flatten_inputs"],
    )

    residual: bool = common_fields.ResidualField()

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=COMBINER_METADATA["concat"]["use_bias"],
    )

    bias_initializer: Union[str, Dict] = common_fields.BiasInitializerField()

    weights_initializer: Union[str, Dict] = common_fields.WeightsInitializerField()

    num_fc_layers: int = common_fields.NumFCLayersField()

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of a fully connected layer.",
        parameter_metadata=COMBINER_METADATA["concat"]["output_size"],
    )

    norm: Optional[str] = common_fields.NormField()

    norm_params: Optional[dict] = common_fields.NormParamsField()

    fc_layers: Optional[List[Dict[str, Any]]] = common_fields.FCLayersField()
