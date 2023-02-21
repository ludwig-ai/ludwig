from typing import Any, Dict, List, Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class ComparatorCombinerConfig(BaseCombinerConfig):
    """Parameters for comparator combiner."""

    def __post_init__(self):
        if self.num_fc_layers == 0 and self.fc_layers is None:
            raise ConfigValidationError(
                "`combiner.type=comparator` requires at least one fully connected layer. "
                "Set `num_fc_layers > 0` or `fc_layers`."
            )

    @staticmethod
    def module_name():
        return "ComparatorCombiner"

    type: str = schema_utils.ProtectedString(
        "comparator",
        description=COMBINER_METADATA["ComparatorCombiner"]["type"].long_description,
    )

    entity_1: List[str] = schema_utils.List(
        default=None,
        description=(
            "The list of input feature names `[feature_1, feature_2, ...]` constituting the first entity to compare. "
            "*Required*."
        ),
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["entity_1"],
    )

    entity_2: List[str] = schema_utils.List(
        default=None,
        description=(
            "The list of input feature names `[feature_1, feature_2, ...]` constituting the second entity to compare. "
            "*Required*."
        ),
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["entity_2"],
    )

    dropout: float = common_fields.DropoutField()

    activation: str = schema_utils.ActivationOptions(default="relu")

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["use_bias"],
    )

    bias_initializer: Union[str, Dict] = common_fields.BiasInitializerField()

    weights_initializer: Union[str, Dict] = common_fields.WeightsInitializerField()

    num_fc_layers: int = common_fields.NumFCLayersField(default=1)

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of a fully connected layer.",
        parameter_metadata=COMBINER_METADATA["ComparatorCombiner"]["output_size"],
    )

    norm: Optional[str] = common_fields.NormField()

    norm_params: Optional[dict] = common_fields.NormParamsField()

    fc_layers: Optional[List[Dict[str, Any]]] = common_fields.FCLayersField()
