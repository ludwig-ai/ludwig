from marshmallow_dataclass import dataclass

from ludwig.constants import MEAN_SQUARED_ERROR, NUMBER
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class NumberInputFeatureConfig(BaseInputFeatureConfig):
    """NumberInputFeatureConfig is a dataclass that configures the parameters used for a number input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=NUMBER)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type="number",
        default="passthrough",
    )


@dataclass
class NumberOutputFeatureConfig(BaseOutputFeatureConfig):
    """NumberOutputFeatureConfig is a dataclass that configures the parameters used for a category output
    feature."""

    loss: dict = schema_utils.Dict(
        default={
            "type": MEAN_SQUARED_ERROR,
            "weight": 1,
        },
        description="A dictionary containing a loss type and its hyper-parameters.",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=NUMBER,
        default="regressor",
    )
