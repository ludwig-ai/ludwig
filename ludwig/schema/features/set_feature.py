from marshmallow_dataclass import dataclass

from ludwig.constants import SET, SIGMOID_CROSS_ENTROPY
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


@input_config_registry.register(SET)
@dataclass
class SetInputFeatureConfig(BaseInputFeatureConfig):
    """SetInputFeatureConfig is a dataclass that configures the parameters used for a set input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SET)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=SET,
        default="embed",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )


@output_config_registry.register(SET)
@dataclass
class SetOutputFeatureConfig(BaseOutputFeatureConfig):
    """SetOutputFeatureConfig is a dataclass that configures the parameters used for a set output feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="set_output")

    loss: BaseLossConfig = LossDataclassField(
        feature_type=SET,
        default=SIGMOID_CROSS_ENTROPY,
    )

    threshold: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description="The threshold used to convert output probabilities to predictions. Tokens with predicted"
        "probabilities greater than or equal to threshold are predicted to be in the output set (True).",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=SET,
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
