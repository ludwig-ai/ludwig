from marshmallow_dataclass import dataclass

from ludwig.constants import SEQUENCE, SEQUENCE_SOFTMAX_CROSS_ENTROPY
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class SequenceInputFeatureConfig(BaseInputFeatureConfig):
    """SequenceInputFeatureConfig is a dataclass that configures the parameters used for a sequence input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SEQUENCE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=SEQUENCE,
        default="embed",
    )


@dataclass
class SequenceOutputFeatureConfig(BaseOutputFeatureConfig):
    """SequenceOutputFeatureConfig is a dataclass that configures the parameters used for a sequence output
    feature."""

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
        feature_type=SEQUENCE,
        default="generator",
    )
