from marshmallow_dataclass import dataclass

from ludwig.constants import CATEGORY, SOFTMAX_CROSS_ENTROPY
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class CategoryInputFeatureConfig(BaseInputFeatureConfig):
    """CategoryInputFeatureConfig is a dataclass that configures the parameters used for a category input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=CATEGORY)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=CATEGORY,
        default="dense",
    )


@dataclass
class CategoryOutputFeatureConfig(BaseOutputFeatureConfig):
    """CategoryOutputFeatureConfig is a dataclass that configures the parameters used for a category output
    feature."""

    loss: dict = schema_utils.Dict(
        default={
            "type": SOFTMAX_CROSS_ENTROPY,
            "class_weights": 1,
            "robust_lambda": 0,
            "confidence_penalty": 0,
            "class_similarities_temperature": 0,
            "weight": 1,
        },
        description="A dictionary containing a loss type and its hyper-parameters.",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=CATEGORY,
        default="classifier",
    )

    top_k: int = schema_utils.NonNegativeInteger(
        default=3,
        description="Determines the parameter k, the number of categories to consider when computing the top_k "
        "measure. It computes accuracy but considering as a match if the true category appears in the "
        "first k predicted categories ranked by decoder's confidence.",
    )
