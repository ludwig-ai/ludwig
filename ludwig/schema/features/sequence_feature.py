from marshmallow_dataclass import dataclass

from ludwig.constants import LOSS, SEQUENCE, SEQUENCE_SOFTMAX_CROSS_ENTROPY
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
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY


@input_config_registry.register(SEQUENCE)
@dataclass(repr=False)
class SequenceInputFeatureConfig(BaseInputFeatureConfig):
    """SequenceInputFeatureConfig is a dataclass that configures the parameters used for a sequence input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SEQUENCE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=SEQUENCE,
        default="embed",
    )


@output_config_registry.register(SEQUENCE)
@dataclass(repr=False)
class SequenceOutputFeatureConfig(BaseOutputFeatureConfig):
    """SequenceOutputFeatureConfig is a dataclass that configures the parameters used for a sequence output
    feature."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=SEQUENCE,
        default="generator",
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [LOSS],
        default=LOSS,
        description="Internal only use parameter: default validation metric for sequence output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=SEQUENCE,
        default=SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="sequence_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )
