from marshmallow_dataclass import dataclass

from ludwig.constants import SEQUENCE
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.decoders.base import BaseDecoderConfig


@dataclass
class SequenceInputFeatureConfig(BaseInputFeatureConfig):
    """SequenceInputFeatureConfig is a dataclass that configures the parameters used for a sequence input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SEQUENCE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=SEQUENCE,
        default='embed',
    )


@dataclass
class SequenceOutputFeatureConfig(BaseOutputFeatureConfig):
    """SequenceOutputFeatureConfig is a dataclass that configures the parameters used for a sequence output
    feature."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=SEQUENCE,
        default='generator',
    )
