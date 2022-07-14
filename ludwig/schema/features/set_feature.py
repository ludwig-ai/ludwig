from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import SET
from ludwig.decoders.registry import get_decoder_classes
from ludwig.encoders.registry import get_encoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class SetInputFeatureConfig(BaseInputFeatureConfig):
    """SetInputFeatureConfig is a dataclass that configures the parameters used for a set input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SET)

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(SET).keys()),
        default="embed",
        description="Encoder to use for this set feature.",
    )


@dataclass
class SetOutputFeatureConfig(BaseOutputFeatureConfig):
    """SetOutputFeatureConfig is a dataclass that configures the parameters used for a set output feature."""

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(SET).keys()),
        default="classifier",
        allow_none=True,
        description="Decoder to use for this set feature.",
    )
