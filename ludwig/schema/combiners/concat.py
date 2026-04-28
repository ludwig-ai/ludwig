from typing import Any

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import register_combiner_config
from ludwig.schema.metadata import COMBINER_METADATA


@DeveloperAPI
@register_combiner_config("concat")
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

    bias_initializer: str | dict = common_fields.BiasInitializerField()

    weights_initializer: str | dict = common_fields.WeightsInitializerField()

    num_fc_layers: int = common_fields.NumFCLayersField()

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output size of a fully connected layer.",
        parameter_metadata=COMBINER_METADATA["concat"]["output_size"],
    )

    norm: str | None = common_fields.NormField()

    norm_params: dict | None = common_fields.NormParamsField()

    fc_layers: list[dict[str, Any]] | None = common_fields.FCLayersField()

    batch_ensemble: bool = schema_utils.Boolean(
        default=False,
        description=(
            "Whether to use BatchEnsemble (TabM-style) for parameter-efficient ensembling. "
            "Adds per-member rank-1 scaling vectors to the output layer, providing "
            "ensemble-level performance at single-model cost."
        ),
    )

    num_ensemble_members: int = schema_utils.PositiveInteger(
        default=4,
        description="Number of ensemble members when batch_ensemble is enabled.",
    )
