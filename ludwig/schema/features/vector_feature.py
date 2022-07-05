from typing import Optional
from ludwig.constants import VECTOR

from marshmallow_dataclass import dataclass

from ludwig.encoders.registry import get_encoder_classes
from ludwig.decoders.registry import get_decoder_classes

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class VectorInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    VectorInputFeatureConfig is a dataclass that configures the parameters used for a vector input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type=VECTOR
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(VECTOR).keys()),
        default="dense",
        description="Encoder to use for this vector feature.",
    )


@dataclass
class VectorOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    VectorOutputFeatureConfig is a dataclass that configures the parameters used for a vector output feature.
    """

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(VECTOR).keys()),
        default="projector",
        description="Decoder to use for this vector feature.",
    )

