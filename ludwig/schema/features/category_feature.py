from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import CATEGORY
from ludwig.decoders.registry import get_decoder_classes
from ludwig.encoders.registry import get_encoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class CategoryInputFeatureConfig(BaseInputFeatureConfig):
    """CategoryInputFeature is a dataclass that configures the parameters used for a category input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=CATEGORY)

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(CATEGORY).keys()),
        default="dense",
        description="Encoder to use for this category feature.",
    )


@dataclass
class CategoryOutputFeatureConfig(BaseOutputFeatureConfig):
    """CategoryOutputFeature is a dataclass that configures the parameters used for a category output feature."""

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(CATEGORY).keys()),
        default="classifier",
        allow_none=True,
        description="Decoder to use for this category feature.",
    )
