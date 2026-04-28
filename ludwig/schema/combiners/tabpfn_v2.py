"""Schema for the TabPFN v2 combiner (Phase 6.5)."""

from __future__ import annotations

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import register_combiner_config


@DeveloperAPI
@register_combiner_config("tabpfn_v2")
class TabPFNV2CombinerConfig(BaseCombinerConfig):
    """TabPFN v2 foundation-model combiner.

    Wraps the pretrained TabPFN v2 (Hollmann et al., 2022 / 2025) as the ECD fusion
    block. Best suited for small tabular datasets (<=10k rows) where in-context
    learning outperforms gradient-based fine-tuning. Requires the optional ``tabpfn``
    Python package — ``pip install tabpfn``.
    """

    type: str = schema_utils.ProtectedString(
        "tabpfn_v2",
        description="TabPFN v2 foundation-model combiner for tabular data.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Width of the learnable projection head applied to TabPFN's encoder output.",
    )

    tabpfn_hidden_size: int = schema_utils.PositiveInteger(
        default=512,
        description="TabPFN v2's internal hidden width. The v2 default is 512; only change this if loading a variant.",
    )

    n_estimators: int = schema_utils.PositiveInteger(
        default=4,
        description=(
            "Number of TabPFN ensemble members to use during prediction. Higher values improve accuracy at the cost of "
            "inference latency."
        ),
    )

    device: str = schema_utils.StringOptions(
        options=["auto", "cpu", "cuda"],
        default="auto",
        allow_none=False,
        description="Device used for TabPFN inference. 'auto' picks CUDA if available.",
    )
