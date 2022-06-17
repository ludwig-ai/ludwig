from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class CategoryInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """CategoryInputFeature is a dataclass that configures the parameters used for a category input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='category'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["passthrough", "dense", "sparse"],
        default="dense",
        description="Encoder to use for this category feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class CategoryOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """CategoryOutputFeature is a dataclass that configures the parameters used for a category output feature."""

    decoder: Optional[str] = schema_utils.StringOptions(
        ["classifier"],
        default="classifier",
        allow_none=True,
        description="Decoder to use for this category feature.",
    )
