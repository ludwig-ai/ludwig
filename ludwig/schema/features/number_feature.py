from marshmallow_dataclass import dataclass

from ludwig.constants import NUMBER
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.decoders.base import BaseDecoderConfig


@dataclass
class NumberInputFeatureConfig(BaseInputFeatureConfig):
    """NumberInputFeature is a dataclass that configures the parameters used for a number input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=NUMBER)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type='number',
        default='passthrough',
    )


@dataclass
class NumberOutputFeatureConfig(BaseOutputFeatureConfig):

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=NUMBER,
        default='regressor',
    )
