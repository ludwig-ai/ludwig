from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils
from ludwig.schema.combiners.base_combiner_config import BaseCombinerConfig


@dataclass
class TabNetCombinerConfig(BaseCombinerConfig):
    """Parameters for tabnet combiner."""

    size: int = utils.PositiveInteger(default=32, description="`N_a` in the paper.")

    output_size: int = utils.PositiveInteger(
        default=32, description="Output size of a fully connected layer. `N_d` in the paper"
    )

    num_steps: int = utils.NonNegativeInteger(
        default=1,
        description=(
            "Number of steps / repetitions of the the attentive transformer and feature transformer computations. "
            "`N_steps` in the paper"
        ),
    )

    num_total_blocks: int = utils.NonNegativeInteger(
        default=4, description="Total number of feature transformer block at each step"
    )

    num_shared_blocks: int = utils.NonNegativeInteger(
        default=2, description="Number of shared feature transformer blocks across the steps"
    )

    relaxation_factor: float = utils.FloatRange(
        default=1.5,
        description=(
            "Factor that influences how many times a feature should be used across the steps of computation. a value of"
            " 1 implies it each feature should be use once, a higher value allows for multiple usages. `gamma` in the "
            "paper"
        ),
    )

    bn_epsilon: float = utils.FloatRange(default=1e-3, description="Epsilon to be added to the batch norm denominator.")

    bn_momentum: float = utils.FloatRange(default=0.7, description="Momentum of the batch norm. `m_B` in the paper.")

    bn_virtual_bs: Optional[int] = utils.PositiveInteger(
        default=None,
        description=(
            "Size of the virtual batch size used by ghost batch norm. If null, regular batch norm is used instead. "
            "`B_v` from the paper"
        ),
    )

    sparsity: float = utils.FloatRange(
        default=1e-5, description="Multiplier of the sparsity inducing loss. `lambda_sparse` in the paper"
    )

    entmax_mode: str = utils.StringOptions(
        ["entmax15", "sparsemax", "constant", "adaptive"], default="sparsemax", description="TODO: Document parameters."
    )

    entmax_alpha: float = utils.FloatRange(
        default=1.5, min=1, max=2, description="TODO: Document parameters."
    )  # 1 corresponds to softmax, 2 is sparsemax.

    dropout: float = utils.FloatRange(default=0.0, min=0, max=1, description="Dropout rate for the transformer block.")
