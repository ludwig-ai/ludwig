from marshmallow_dataclass import dataclass

from ludwig.constants import BINARY, BINARY_WEIGHTED_CROSS_ENTROPY
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class BinaryInputFeatureConfig(BaseInputFeatureConfig):
    """BinaryInputFeatureConfig is a dataclass that configures the parameters used for a binary input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=BINARY)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=BINARY,
        default="passthrough",
    )


@dataclass
class BinaryOutputFeatureConfig(BaseOutputFeatureConfig):
    """BinaryOutputFeatureConfig is a dataclass that configures the parameters used for a binary output feature."""

    loss: dict = schema_utils.Dict(
        default={
            "type": BINARY_WEIGHTED_CROSS_ENTROPY,
            "robust_lambda": 0,
            "confidence_penalty": 0,
            "positive_class_weight": None,
        },
        description="A dictionary containing a loss type and its hyper-parameters.",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=BINARY,
        default="regressor",
    )
