from marshmallow_dataclass import dataclass

from ludwig.constants import BINARY
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.decoders.base import BaseDecoderConfig


@dataclass
class BinaryInputFeatureConfig(BaseInputFeatureConfig):
    """BinaryInputFeature is a dataclass that configures the parameters used for a binary input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=BINARY)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=BINARY,
        default='passthrough',
    )


@dataclass
class BinaryOutputFeatureConfig(BaseOutputFeatureConfig):
    """BinaryOutputFeature is a dataclass that configures the parameters used for a binary output feature."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=BINARY,
        default='regressor',
    )
