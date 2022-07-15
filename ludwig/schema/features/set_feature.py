from marshmallow_dataclass import dataclass

from ludwig.constants import SET
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.decoders.base import BaseDecoderConfig


@dataclass
class SetInputFeatureConfig(BaseInputFeatureConfig):
    """SetInputFeatureConfig is a dataclass that configures the parameters used for a set input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SET)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=SET,
        default='embed',
    )


@dataclass
class SetOutputFeatureConfig(BaseOutputFeatureConfig):
    """SetOutputFeatureConfig is a dataclass that configures the parameters used for a set output feature."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=SET,
        default='classifier',
    )
