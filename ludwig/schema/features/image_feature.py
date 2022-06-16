from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class ImageInputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseFeatureConfig):
    """
    ImageInputFeatureConfig is a dataclass that configures the parameters used for an image input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='image'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["stacked_cnn", "resnet", "mlp_mixer", "vit"],
        default="stacked_cnn",
        description="Encoder to use for this image feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )
