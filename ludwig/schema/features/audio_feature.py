from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import AudioPreprocessingConfig


@dataclass
class AudioInputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseFeatureConfig):
    """AudioFeatureInputFeature is a dataclass that configures the parameters used for an audio input feature."""

    name: str = schema_utils.String(
        default=None,
        allow_none=False,
        description="Name of the feature. Must be unique within the model.",
    )

    type: str = "audio"

    preprocessing: Optional[str] = AudioPreprocessingConfig(
    )
