from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, BINARY_WEIGHTED_CROSS_ENTROPY, MODEL_ECD, MODEL_GBM, ROC_AUC
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
    defaults_config_registry,
    ecd_input_config_registry,
    gbm_input_config_registry,
    input_mixin_registry,
    output_config_registry,
    output_mixin_registry,
)
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@input_mixin_registry.register(BINARY)
@ludwig_dataclass
class BinaryInputFeatureConfigMixin(BaseMarshmallowConfig):
    """BinaryInputFeatureConfigMixin is a dataclass that configures the parameters used in both the binary input
    feature and the binary global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=BINARY)


@DeveloperAPI
@ludwig_dataclass
class BinaryInputFeatureConfig(BaseInputFeatureConfig, BinaryInputFeatureConfigMixin):
    """BinaryInputFeatureConfig is a dataclass that configures the parameters used for a binary input feature."""

    encoder: BaseEncoderConfig = None


@DeveloperAPI
@ecd_input_config_registry.register(BINARY)
@ludwig_dataclass
class ECDBinaryInputFeatureConfig(BinaryInputFeatureConfig):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=BINARY,
        default="passthrough",
    )


@DeveloperAPI
@gbm_input_config_registry.register(BINARY)
@ludwig_dataclass
class GBMBinaryInputFeatureConfig(BinaryInputFeatureConfig):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_GBM,
        feature_type=BINARY,
        default="passthrough",
    )


@DeveloperAPI
@output_mixin_registry.register(BINARY)
@ludwig_dataclass
class BinaryOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """BinaryOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the binary output
    feature and the binary global defaults section of the Ludwig Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=BINARY,
        default="regressor",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=BINARY,
        default=BINARY_WEIGHTED_CROSS_ENTROPY,
    )


@DeveloperAPI
@output_config_registry.register(BINARY)
@ludwig_dataclass
class BinaryOutputFeatureConfig(BaseOutputFeatureConfig, BinaryOutputFeatureConfigMixin):
    """BinaryOutputFeatureConfig is a dataclass that configures the parameters used for a binary output feature."""

    calibration: bool = schema_utils.Boolean(
        default=False,
        description="Calibrate the model's output probabilities using temperature scaling.",
        parameter_metadata=FEATURE_METADATA[BINARY]["calibration"],
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [ROC_AUC],
        default=ROC_AUC,
        description="Internal only use parameter: default validation metric for binary output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[BINARY]["dependencies"],
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="binary_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[BINARY]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[BINARY]["reduce_input"],
    )

    threshold: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description="The threshold used to convert output probabilities to predictions. Predicted probabilities greater"
        "than or equal to threshold are mapped to True.",
        parameter_metadata=FEATURE_METADATA[BINARY]["threshold"],
    )


@DeveloperAPI
@defaults_config_registry.register(BINARY)
@ludwig_dataclass
class BinaryDefaultsConfig(BinaryInputFeatureConfigMixin, BinaryOutputFeatureConfigMixin):
    # NOTE(travis): defaults use ECD input feature as it contains all the encoders
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=BINARY,
        default="passthrough",
    )
