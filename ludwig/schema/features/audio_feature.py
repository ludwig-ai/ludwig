from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class AudioInputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseFeatureConfig):
    """AudioFeatureInputFeature is a dataclass that configures the parameters used for an audio input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='audio'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["passthrough", "parallel_cnn", "stacked_cnn", "stacked_parallel_cnn", "rnn", "cnnrnn"],
        default="parallel_cnn",
        description="Encoder to use for this audio feature.",
    )

    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )

