from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TIMESERIES
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config


@DeveloperAPI
@register_encoder_config("patchtst", [TIMESERIES])
class PatchTSTConfig(BaseEncoderConfig):
    """PatchTST encoder config.

    From: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (Nie et al., 2023).
    Splits the time series into fixed-length patches, projects each patch to an embedding, then applies
    a standard Transformer encoder. Channel-independent: each variate is processed separately.
    """

    @staticmethod
    def module_name():
        return "PatchTSTEncoder"

    type: str = schema_utils.ProtectedString("patchtst")

    patch_size: int = schema_utils.PositiveInteger(
        default=16,
        description="Length of each patch (number of time steps per patch).",
    )

    patch_stride: int = schema_utils.PositiveInteger(
        default=8,
        description="Stride between consecutive patches. Use patch_stride < patch_size for overlapping patches.",
    )

    d_model: int = schema_utils.PositiveInteger(
        default=128,
        description="Transformer hidden dimension.",
    )

    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=3,
        description="Number of Transformer encoder layers.",
    )

    ffn_dim: int = schema_utils.PositiveInteger(
        default=256,
        description="Feed-forward network hidden dimension inside Transformer blocks.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0.0,
        max=1.0,
        description="Dropout rate.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output embedding size after pooling patches.",
    )

    reduce_output: str = schema_utils.StringOptions(
        options=["mean", "last", "first"],
        default="mean",
        description=(
            "How to aggregate patch representations: 'mean' pools all patches, "
            "'last' uses the last, 'first' uses the first."
        ),
    )


@DeveloperAPI
@register_encoder_config("nbeats", [TIMESERIES])
class NBEATSConfig(BaseEncoderConfig):
    """N-BEATS encoder config.

    From: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
    (Oreshkin et al., 2020). A pure MLP architecture with doubly residual stacking: each block produces
    a backcast (reconstruction of input) and a forecast, which are summed across blocks. The encoder
    returns a fixed-size representation of the backcast residual suitable for downstream use.
    """

    @staticmethod
    def module_name():
        return "NBEATSEncoder"

    type: str = schema_utils.ProtectedString("nbeats")

    num_stacks: int = schema_utils.PositiveInteger(
        default=2,
        description="Number of N-BEATS stacks.",
    )

    num_blocks: int = schema_utils.PositiveInteger(
        default=3,
        description="Number of blocks per stack.",
    )

    num_layers: int = schema_utils.PositiveInteger(
        default=4,
        description="Number of fully-connected layers per block.",
    )

    layer_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Width of fully-connected layers in each block.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the final output representation.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description="Dropout rate inside blocks.",
    )
