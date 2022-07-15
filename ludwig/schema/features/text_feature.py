from marshmallow_dataclass import dataclass

from ludwig.constants import TEXT
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.decoders.base import BaseDecoderConfig


@dataclass
class TextInputFeatureConfig(BaseInputFeatureConfig):
    """TextInputFeatureConfig is a dataclass that configures the parameters used for a text input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=TEXT)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=TEXT,
        default='parallel_cnn',
    )


@dataclass
class TextOutputFeatureConfig(BaseOutputFeatureConfig):
    """TextOutputFeatureConfig is a dataclass that configures the parameters used for a text output feature."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=TEXT,
        default='generator',
    )
