from marshmallow_dataclass import dataclass

from ludwig.constants import VECTOR
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.decoders.base import BaseDecoderConfig


@dataclass
class VectorInputFeatureConfig(BaseInputFeatureConfig):
    """VectorInputFeatureConfig is a dataclass that configures the parameters used for a vector input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=VECTOR)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=VECTOR,
        default='dense',
    )


@dataclass
class VectorOutputFeatureConfig(BaseOutputFeatureConfig):
    """VectorOutputFeatureConfig is a dataclass that configures the parameters used for a vector output feature."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=VECTOR,
        default='projector',
    )
