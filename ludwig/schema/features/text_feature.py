from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import LOSS, MODEL_ECD, SEQUENCE_SOFTMAX_CROSS_ENTROPY, TEXT
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
    input_mixin_registry,
    output_config_registry,
    output_mixin_registry,
)
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@input_mixin_registry.register(TEXT)
@ludwig_dataclass
class TextInputFeatureConfigMixin(BaseMarshmallowConfig):
    """TextInputFeatureConfigMixin is a dataclass that configures the parameters used in both the text input
    feature and the text global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=TEXT)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=TEXT,
        default="parallel_cnn",
    )


@DeveloperAPI
@ecd_input_config_registry.register(TEXT)
@ludwig_dataclass
class TextInputFeatureConfig(BaseInputFeatureConfig, TextInputFeatureConfigMixin):
    """TextInputFeatureConfig is a dataclass that configures the parameters used for a text input feature."""

    pass


@DeveloperAPI
@output_mixin_registry.register(TEXT)
@ludwig_dataclass
class TextOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """TextOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the text output
    feature and the text global defaults section of the Ludwig Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=TEXT,
        default="generator",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=TEXT,
        default=SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    )


@DeveloperAPI
@output_config_registry.register(TEXT)
@ludwig_dataclass
class TextOutputFeatureConfig(BaseOutputFeatureConfig, TextOutputFeatureConfigMixin):
    """TextOutputFeatureConfig is a dataclass that configures the parameters used for a text output feature."""

    class_similarities: list = schema_utils.List(
        list,
        default=None,
        description="If not null this parameter is a c x c matrix in the form of a list of lists that contains the "
        "mutual similarity of classes. It is used if `class_similarities_temperature` is greater than 0. ",
        parameter_metadata=FEATURE_METADATA[TEXT]["class_similarities"],
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [LOSS],
        default=LOSS,
        description="Internal only use parameter: default validation metric for binary output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[TEXT]["dependencies"],
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="text_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[TEXT]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[TEXT]["reduce_input"],
    )


@DeveloperAPI
@defaults_config_registry.register(TEXT)
@ludwig_dataclass
class TextDefaultsConfig(TextInputFeatureConfigMixin, TextOutputFeatureConfigMixin):
    loss: BaseLossConfig = LossDataclassField(
        feature_type=TEXT,
        default=SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    )
