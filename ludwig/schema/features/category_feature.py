from marshmallow_dataclass import dataclass

from ludwig.constants import CATEGORY
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.decoders.base import BaseDecoderConfig


@dataclass
class CategoryInputFeatureConfig(BaseInputFeatureConfig):
    """CategoryInputFeature is a dataclass that configures the parameters used for a category input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=CATEGORY)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=CATEGORY,
        default='dense',
    )


@dataclass
class CategoryOutputFeatureConfig(BaseOutputFeatureConfig):
    """CategoryOutputFeature is a dataclass that configures the parameters used for a category output feature."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=CATEGORY,
        default='classifier',
    )
