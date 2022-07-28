from marshmallow_dataclass import dataclass

from ludwig.constants import SEQUENCE_SOFTMAX_CROSS_ENTROPY, TEXT
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class TextInputFeatureConfig(BaseInputFeatureConfig):
    """TextInputFeatureConfig is a dataclass that configures the parameters used for a text input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=TEXT)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=TEXT,
        default="parallel_cnn",
    )


@dataclass
class TextOutputFeatureConfig(BaseOutputFeatureConfig):
    """TextOutputFeatureConfig is a dataclass that configures the parameters used for a text output feature."""

    loss: dict = schema_utils.Dict(
        default={
            "type": SEQUENCE_SOFTMAX_CROSS_ENTROPY,
            "class_weights": 1,
            "robust_lambda": 0,
            "confidence_penalty": 0,
            "class_similarities_temperature": 0,
            "weight": 1,
            "unique": False,
        },
        description="A dictionary containing a loss type and its hyper-parameters.",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=TEXT,
        default="generator",
    )
