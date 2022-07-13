from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import NUMBER
from ludwig.decoders.registry import get_decoder_classes
from ludwig.encoders.registry import get_encoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class NumberInputFeatureConfig(BaseInputFeatureConfig):
    """NumberInputFeature is a dataclass that configures the parameters used for a number input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=NUMBER)

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(NUMBER).keys()),
        default="passthrough",
        description="Encoder to use for this number feature.",
    )


@dataclass
class NumberOutputFeatureConfig(BaseOutputFeatureConfig):

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(NUMBER).keys()),
        default="regressor",
        allow_none=True,
        description="Decoder to use for this number feature.",
    )
