from typing import Any, Dict, List, Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class CommonTransformerConfig:
    """Common transformer parameter values."""

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0,
        max=1,
        description="Dropout rate for the transformer block.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["dropout"],
    )

    transformer_output_size: int = schema_utils.NonNegativeInteger(
        default=256,
        description="Size of the fully connected layer after self attention in the transformer block. This is usually "
        "the same as `hidden_size` and `embedding_size`.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["transformer_output_size"],
    )

    hidden_size: int = schema_utils.NonNegativeInteger(
        default=256,
        description="The number of hidden units of the TransformerStack as well as the dimension that each incoming "
        "input feature is projected to before feeding to the TransformerStack.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["hidden_size"],
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=1,
        description="The number of transformer layers.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["num_layers"],
    )

    num_heads: int = schema_utils.NonNegativeInteger(
        default=8,
        description="Number of heads of the self attention in the transformer block.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["num_heads"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["use_bias"],
    )

    bias_initializer: Union[str, Dict] = common_fields.BiasInitializerField()

    weights_initializer: Union[str, Dict] = common_fields.WeightsInitializerField()

    # TODO(#1673): Add conditional logic for fields like this one:
    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The number of stacked fully connected layers (only applies if `reduce_output` is not null).",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["num_fc_layers"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of a fully connected layer.",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["output_size"],
    )

    norm: Optional[str] = common_fields.NormField()

    norm_params: Optional[dict] = common_fields.NormParamsField()

    fc_layers: Optional[List[Dict[str, Any]]] = common_fields.FCLayersField()

    fc_dropout: float = common_fields.DropoutField()

    fc_activation: str = schema_utils.ActivationOptions(
        default="relu",
        parameter_metadata=COMBINER_METADATA["TransformerCombiner"]["fc_activation"],
    )

    fc_residual: bool = common_fields.ResidualField()
