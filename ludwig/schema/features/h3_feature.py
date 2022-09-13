from marshmallow_dataclass import dataclass

from ludwig.constants import H3
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import input_config_registry


@input_config_registry.register(H3)
@dataclass
class H3InputFeatureConfig(BaseInputFeatureConfig):
    """H3InputFeatureConfig is a dataclass that configures the parameters used for an h3 input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=H3)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=H3,
        default="embed",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )
