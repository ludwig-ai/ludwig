from marshmallow_dataclass import dataclass

from ludwig.constants import CATEGORY, SOFTMAX_CROSS_ENTROPY
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.features.loss.loss import BaseLossConfig
from ludwig.schema.features.loss.utils import LossDataclassField
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import input_config_registry, output_config_registry


@input_config_registry.register(CATEGORY)
@dataclass
class CategoryInputFeatureConfig(BaseInputFeatureConfig):
    """CategoryInputFeatureConfig is a dataclass that configures the parameters used for a category input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=CATEGORY)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=CATEGORY,
        default="dense",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )


@output_config_registry.register(CATEGORY)
@dataclass
class CategoryOutputFeatureConfig(BaseOutputFeatureConfig):
    """CategoryOutputFeatureConfig is a dataclass that configures the parameters used for a category output
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="category_output")

    loss: BaseLossConfig = LossDataclassField(
        feature_type=CATEGORY,
        default=SOFTMAX_CROSS_ENTROPY,
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=CATEGORY,
        default="classifier",
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
    )

    top_k: int = schema_utils.NonNegativeInteger(
        default=3,
        description="Determines the parameter k, the number of categories to consider when computing the top_k "
        "measure. It computes accuracy but considering as a match if the true category appears in the "
        "first k predicted categories ranked by decoder's confidence.",
    )

    calibration: bool = schema_utils.Boolean(
        default=False,
        description="Calibrate the model's output probabilities using temperature scaling.",
    )
