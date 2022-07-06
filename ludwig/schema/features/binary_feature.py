from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import BINARY
from ludwig.decoders.registry import get_decoder_classes
from ludwig.encoders.registry import get_encoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class BinaryInputFeatureConfig(BaseInputFeatureConfig):
    """BinaryInputFeature is a dataclass that configures the parameters used for a binary input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=BINARY)

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(BINARY).keys()),
        default="passthrough",
        description="Encoder to use for this binary feature.",
    )


@dataclass
class BinaryOutputFeatureConfig(BaseOutputFeatureConfig):
    """BinaryOutputFeature is a dataclass that configures the parameters used for a binary output feature."""

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(BINARY).keys()),
        default="regressor",
        allow_none=True,
        description="Decoder to use for this binary feature.",
    )
