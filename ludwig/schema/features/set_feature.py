from marshmallow_dataclass import dataclass

from ludwig.constants import JACCARD, SET, SIGMOID_CROSS_ENTROPY
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
from ludwig.schema.features.utils import (
    input_config_registry,
    input_mixin_registry,
    output_config_registry,
    output_mixin_registry,
)
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
from ludwig.schema.utils import BaseMarshmallowConfig


@input_mixin_registry.register(SET)
@dataclass
class SetInputFeatureConfigMixin(BaseMarshmallowConfig):
    """SetInputFeatureConfigMixin is a dataclass that configures the parameters used in both the set input feature
    and the set global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SET)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=SET,
        default="embed",
    )


@input_config_registry.register(SET)
@dataclass(repr=False)
class SetInputFeatureConfig(BaseInputFeatureConfig, SetInputFeatureConfigMixin):
    """SetInputFeatureConfig is a dataclass that configures the parameters used for a set input feature."""

    pass


@output_mixin_registry.register(SET)
@dataclass
class SetOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """SetOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the set output
    feature and the set global defaults section of the Ludwig Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=SET,
        default="classifier",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=SET,
        default=SIGMOID_CROSS_ENTROPY,
    )


@output_config_registry.register(SET)
@dataclass(repr=False)
class SetOutputFeatureConfig(BaseOutputFeatureConfig, SetOutputFeatureConfigMixin):
    """SetOutputFeatureConfig is a dataclass that configures the parameters used for a set output feature."""

    default_validation_metric: str = schema_utils.StringOptions(
        [JACCARD],
        default=JACCARD,
        description="Internal only use parameter: default validation metric for set output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="set_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )

    threshold: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description="The threshold used to convert output probabilities to predictions. Tokens with predicted"
        "probabilities greater than or equal to threshold are predicted to be in the output set (True).",
    )
