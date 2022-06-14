from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import CategoryPreprocessingConfig


@dataclass
class CategoryInputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseInputFeatureConfig):
    """CategoryInputFeature is a dataclass that configures the parameters used for a category input feature."""

    name: str = schema_utils.String(
        default=None,
        allow_none=False,
        description="Name of the feature. Must be unique within the model.",
    )

    type: str = "category"

    preprocessing: Optional[str] = CategoryPreprocessingConfig(
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
class CategoryOutputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseInputFeatureConfig):
    """CategoryOutputFeature is a dataclass that configures the parameters used for a category output feature."""

    name: str = schema_utils.String(
        default=None,
        allow_none=False,
        description="Name of the feature. Must be unique within the model.",
    )

    type: str = "category"
