from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig


@dataclass
class TabNetCombinerConfig(BaseCombinerConfig):
    """Parameters for tabnet combiner."""

    size: int = schema_utils.PositiveInteger(default=32, description="`N_a` in the paper.")

    output_size: int = schema_utils.PositiveInteger(
        default=128, description="Output size of a fully connected layer. `N_d` in the paper"
    )

    num_steps: int = schema_utils.NonNegativeInteger(
        default=3,
        description=(
            "Number of steps / repetitions of the the attentive transformer and feature transformer computations. "
            "`N_steps` in the paper"
        ),
    )

    num_total_blocks: int = schema_utils.NonNegativeInteger(
        default=4, description="Total number of feature transformer block at each step"
    )

    num_shared_blocks: int = schema_utils.NonNegativeInteger(
        default=2, description="Number of shared feature transformer blocks across the steps"
    )

    relaxation_factor: float = schema_utils.FloatRange(
        default=1.5,
        description=(
            "Factor that influences how many times a feature should be used across the steps of computation. a value of"
            " 1 implies it each feature should be use once, a higher value allows for multiple usages. `gamma` in the "
            "paper"
        ),
    )

    bn_epsilon: float = schema_utils.FloatRange(
        default=1e-3, description="Epsilon to be added to the batch norm denominator."
    )

    bn_momentum: float = schema_utils.FloatRange(
        default=0.95, description="Momentum of the batch norm. `m_B` in the paper."
    )

    bn_virtual_bs: Optional[int] = schema_utils.PositiveInteger(
        default=1024,
        allow_none=True,
        description=(
            "Size of the virtual batch size used by ghost batch norm. If null, regular batch norm is used instead. "
            "`B_v` from the paper"
        ),
    )

    sparsity: float = schema_utils.FloatRange(
        default=1e-4, description="Multiplier of the sparsity inducing loss. `lambda_sparse` in the paper"
    )

    entmax_mode: str = schema_utils.StringOptions(
        ["entmax15", "sparsemax", "constant", "adaptive"], default="sparsemax", description=""
    )

    entmax_alpha: float = schema_utils.FloatRange(
        default=1.5, min=1, max=2, description=""
    )  # 1 corresponds to softmax, 2 is sparsemax.

    dropout: float = schema_utils.FloatRange(
        default=0.05, min=0, max=1, description="Dropout rate for the transformer block."
    )
