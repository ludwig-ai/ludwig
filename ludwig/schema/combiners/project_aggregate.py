from typing import Any, Dict, List, Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class ProjectAggregateCombinerConfig(BaseCombinerConfig):
    @staticmethod
    def module_name():
        return "ProjectAggregateCombiner"

    type: str = schema_utils.ProtectedString(
        "project_aggregate",
        description=COMBINER_METADATA["ProjectAggregateCombiner"]["type"].long_description,
    )

    projection_size: int = schema_utils.PositiveInteger(
        default=128,
        description="All combiner inputs are projected to this size before being aggregated.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["projection_size"],
    )

    residual: bool = schema_utils.Boolean(
        default=True,
        description="Whether to add residual skip connection between the fully connected layers in the stack.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["residual"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate to apply to each fully connected layer.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["dropout"],
    )

    activation: str = schema_utils.ActivationOptions(
        default="relu",
        description="Activation to apply to each fully connected layer.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["activation"],
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=2,
        description="Number of fully connected layers after aggregation.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["num_fc_layers"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Output size of each layer of the stack of fully connected layers.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["output_size"],
    )

    norm: Optional[str] = schema_utils.StringOptions(
        ["batch", "layer"],
        default="layer",
        description="Normalization to apply to each projection and fully connected layer.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["norm"],
    )

    norm_params: Optional[dict] = schema_utils.Dict(
        description="Parameters of the normalization to apply to each projection and fully connected layer.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["norm_params"],
    )

    fc_layers: Optional[List[Dict[str, Any]]] = schema_utils.DictList(
        description="Full specification of the fully connected layers after the aggregation. It should be a list of "
        "dict, each dict representing one layer of the fully connected layer stack. ",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["fc_layers"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layers use a bias vector.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["use_bias"],
    )

    bias_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="zeros",
        description="Initializer to use for the bias of the projection and for the fully connected layers.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["bias_initializer"],
    )

    weights_initializer: Union[str, Dict] = schema_utils.InitializerOrDict(
        default="xavier_uniform",
        description="Initializer to use for the weights of the projection and for the fully connected layers.",
        parameter_metadata=COMBINER_METADATA["ProjectAggregateCombiner"]["weights_initializer"],
    )
