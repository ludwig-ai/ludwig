from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.encoders.registry import get_encoder_classes

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class H3InputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    H3InputFeatureConfig is a dataclass that configures the parameters used for an h3 input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='h3'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes('h3').keys()),
        default="embed",
        description="Encoder to use for this h3 feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )
