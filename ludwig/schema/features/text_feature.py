from marshmallow_dataclass import dataclass

from ludwig.constants import SEQUENCE_SOFTMAX_CROSS_ENTROPY, TEXT
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


@input_config_registry.register(TEXT)
@dataclass
class TextInputFeatureConfig(BaseInputFeatureConfig):
    """TextInputFeatureConfig is a dataclass that configures the parameters used for a text input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=TEXT)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=TEXT,
        default="parallel_cnn",
    )


@output_config_registry.register(TEXT)
@dataclass
class TextOutputFeatureConfig(BaseOutputFeatureConfig):
    """TextOutputFeatureConfig is a dataclass that configures the parameters used for a text output feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="text_output")

    loss: BaseLossConfig = LossDataclassField(
        feature_type=TEXT,
        default=SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=TEXT,
        default="generator",
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
