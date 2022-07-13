from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import VECTOR
from ludwig.decoders.registry import get_decoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.utils import BaseEncoderConfig, EncoderDataclassField


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

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(VECTOR).keys()),
        default="projector",
        description="Decoder to use for this vector feature.",
    )
