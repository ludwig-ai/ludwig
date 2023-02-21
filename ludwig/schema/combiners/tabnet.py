from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.metadata import COMBINER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class TabNetCombinerConfig(BaseCombinerConfig):
    """Parameters for tabnet combiner."""

    @staticmethod
    def module_name():
        return "TabNetCombiner"

    type: str = schema_utils.ProtectedString(
        "tabnet",
        description=COMBINER_METADATA["TabNetCombiner"]["type"].long_description,
    )

    size: int = schema_utils.PositiveInteger(
        default=32,
        description="Size of the hidden layers. `N_a` in (Arik and Pfister, 2019).",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["size"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.05,
        min=0,
        max=1,
        description="Dropout rate for the transformer block.",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["dropout"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Output size of a fully connected layer. `N_d` in (Arik and Pfister, 2019).",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["output_size"],
    )

    num_steps: int = schema_utils.NonNegativeInteger(
        default=3,
        description="Number of steps / repetitions of the the attentive transformer and feature transformer "
        "computations. `N_steps` in (Arik and Pfister, 2019).",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["num_steps"],
    )

    num_total_blocks: int = schema_utils.NonNegativeInteger(
        default=4,
        description="Total number of feature transformer blocks at each step.",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["num_total_blocks"],
    )

    num_shared_blocks: int = schema_utils.NonNegativeInteger(
        default=2,
        description="Number of shared feature transformer blocks across the steps.",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["num_shared_blocks"],
    )

    relaxation_factor: float = schema_utils.FloatRange(
        default=1.5,
        description="Factor that influences how many times a feature should be used across the steps of computation. "
        "a value of 1 implies it each feature should be use once, a higher value allows for multiple "
        "usages. `gamma` in (Arik and Pfister, 2019).",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["relaxation_factor"],
    )

    bn_epsilon: float = schema_utils.FloatRange(
        default=1e-3,
        description="Epsilon to be added to the batch norm denominator.",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["bn_epsilon"],
    )

    bn_momentum: float = schema_utils.FloatRange(
        default=0.05,
        description="Momentum of the batch norm. 1 - `m_B` from the TabNet paper.",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["bn_momentum"],
    )

    bn_virtual_bs: Optional[int] = schema_utils.PositiveInteger(
        default=1024,
        allow_none=True,
        description="Size of the virtual batch size used by ghost batch norm. If null, regular batch norm is used "
        "instead. `B_v` from the TabNet paper.",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["bn_virtual_bs"],
    )

    sparsity: float = schema_utils.FloatRange(
        default=1e-4,
        description="Multiplier of the sparsity inducing loss. `lambda_sparse` in (Arik and Pfister, 2019).",
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["sparsity"],
    )

    entmax_mode: str = schema_utils.StringOptions(
        ["entmax15", "sparsemax", "constant", "adaptive"],
        default="sparsemax",
        description=(
            "Entmax is a sparse family of probability mapping which generalizes softmax and sparsemax. "
            "`entmax_mode` controls the sparsity"
        ),
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["entmax_mode"],
    )

    entmax_alpha: float = schema_utils.FloatRange(
        default=1.5,
        min=1,
        max=2,
        description=(
            "Must be a number between 1.0 and 2.0. If entmax_mode is `adaptive`, "
            "`entmax_alpha` is used as the initial value for the learnable parameter. "
            "1 corresponds to softmax, 2 is sparsemax."
        ),
        parameter_metadata=COMBINER_METADATA["TabNetCombiner"]["entmax_alpha"],
    )
