"""Schema configs for number-specific encoders (PLE, Periodic)."""

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import NUMBER
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config


@DeveloperAPI
@register_encoder_config("ple", [NUMBER])
class PLEEncoderConfig(BaseEncoderConfig):
    """Piecewise Linear Encoding for numerical features.

    Computes quantile-based bin edges from training data, then produces a piecewise-linear interpolation vector per
    input value. Most impactful improvement for tabular deep learning accuracy.

    Based on Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning", NeurIPS 2022.
    """

    @staticmethod
    def module_name():
        return "PLEEncoder"

    type: str = schema_utils.ProtectedString(
        "ple",
        description="Piecewise Linear Encoding: quantile-based binning with learned projection.",
    )

    num_bins: int = schema_utils.PositiveInteger(
        default=64,
        description="Number of quantile bins for piecewise linear encoding.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the output embedding.",
    )

    # Internal field: set from training data metadata, not user-configurable
    ple_bin_edges: list[float] | None = schema_utils.List(
        default=None,
        description="[Internal] Quantile bin edges computed from training data. Set automatically.",
    )


@DeveloperAPI
@register_encoder_config("periodic", [NUMBER])
class PeriodicEncoderConfig(BaseEncoderConfig):
    """Periodic encoding for numerical features using learned sinusoidal features.

    Based on Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning", NeurIPS 2022.
    """

    @staticmethod
    def module_name():
        return "PeriodicEncoder"

    type: str = schema_utils.ProtectedString(
        "periodic",
        description="Periodic encoding: learned sin(2*pi*f*x + phi) features with projection.",
    )

    num_frequencies: int = schema_utils.PositiveInteger(
        default=64,
        description="Number of learnable sinusoidal frequencies.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the output embedding.",
    )

    sigma: float = schema_utils.Float(
        default=1.0,
        description="Standard deviation for initializing frequency parameters.",
    )


@DeveloperAPI
@register_encoder_config("bins", [NUMBER])
class BinsEncoderConfig(BaseEncoderConfig):
    """Binning encoder: discretize numbers into equal-width or equal-frequency bins.

    Simpler alternative to PLE. Good for small/medium datasets where PLE may overfit. Each bin gets a learned embedding.
    """

    @staticmethod
    def module_name():
        return "BinsEncoder"

    type: str = schema_utils.ProtectedString(
        "bins",
        description="Binning encoder: discretize numbers into bins with learned embeddings.",
    )

    num_bins: int = schema_utils.PositiveInteger(
        default=32,
        description="Number of bins to discretize numeric values into.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Size of the output embedding per bin.",
    )

    # Internal field: set from training data metadata, not user-configurable
    bins_bin_edges: list[float] | None = schema_utils.List(
        default=None,
        description="[Internal] Bin edges computed from training data. Set automatically.",
    )
