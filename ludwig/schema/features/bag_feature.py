from typing import Optional
from ludwig.constants import BAG

from marshmallow_dataclass import dataclass

from ludwig.encoders.registry import get_encoder_classes

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class BagInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    BagInputFeatureConfig is a dataclass that configures the parameters used for a bag input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type=BAG
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(BAG).keys()),
        default="embed",
        description="Encoder to use for this bag feature.",
    )
