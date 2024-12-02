from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MEAN_SQUARED_ERROR, MODEL_ECD, VECTOR
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
    ecd_defaults_config_registry,
    ecd_input_config_registry,
    ecd_output_config_registry,
    input_mixin_registry,
    output_mixin_registry,
)
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@input_mixin_registry.register(VECTOR)
@ludwig_dataclass
class VectorInputFeatureConfigMixin(BaseMarshmallowConfig):
    """VectorInputFeatureConfigMixin is a dataclass that configures the parameters used in both the vector input
    feature and the vector global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=VECTOR)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=VECTOR,
        default="dense",
    )


@DeveloperAPI
@ecd_input_config_registry.register(VECTOR)
@ludwig_dataclass
class VectorInputFeatureConfig(VectorInputFeatureConfigMixin, BaseInputFeatureConfig):
    """VectorInputFeatureConfig is a dataclass that configures the parameters used for a vector input feature."""

    type: str = schema_utils.ProtectedString(VECTOR)


@DeveloperAPI
@output_mixin_registry.register(VECTOR)
@ludwig_dataclass
class VectorOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """VectorOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the vector output
    feature and the vector global defaults section of the Ludwig Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=VECTOR,
        default="projector",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=VECTOR,
        default=MEAN_SQUARED_ERROR,
    )


@DeveloperAPI
@ecd_output_config_registry.register(VECTOR)
@ludwig_dataclass
class VectorOutputFeatureConfig(VectorOutputFeatureConfigMixin, BaseOutputFeatureConfig):
    """VectorOutputFeatureConfig is a dataclass that configures the parameters used for a vector output feature."""

    type: str = schema_utils.ProtectedString(VECTOR)

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[VECTOR]["dependencies"],
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [MEAN_SQUARED_ERROR],
        default=MEAN_SQUARED_ERROR,
        description="Internal only use parameter: default validation metric for binary output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="vector_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[VECTOR]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[VECTOR]["reduce_input"],
    )

    softmax: bool = schema_utils.Boolean(
        default=False,
        description="Determines whether to apply a softmax at the end of the decoder. This is useful for predicting a "
        "vector of values that sum up to 1 and can be interpreted as probabilities.",
        parameter_metadata=FEATURE_METADATA[VECTOR]["softmax"],
    )

    vector_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The size of the vector. If None, the vector size will be inferred from the data.",
        parameter_metadata=FEATURE_METADATA[VECTOR]["vector_size"],
    )


@DeveloperAPI
@ecd_defaults_config_registry.register(VECTOR)
@ludwig_dataclass
class VectorDefaultsConfig(VectorInputFeatureConfigMixin, VectorOutputFeatureConfigMixin):
    pass
