from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUDIO, MODEL_ECD
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import ecd_defaults_config_registry, ecd_input_config_registry, input_mixin_registry
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@ecd_defaults_config_registry.register(AUDIO)
@input_mixin_registry.register(AUDIO)
@ludwig_dataclass
class AudioInputFeatureConfigMixin(BaseMarshmallowConfig):
    """AudioInputFeatureConfigMixin is a dataclass that configures the parameters used in both the audio input
    feature and the audio global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=AUDIO)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=AUDIO,
        default="parallel_cnn",
    )


@DeveloperAPI
@ecd_input_config_registry.register(AUDIO)
@ludwig_dataclass
class AudioInputFeatureConfig(AudioInputFeatureConfigMixin, BaseInputFeatureConfig):
    """AudioInputFeatureConfig is a dataclass that configures the parameters used for an audio input feature."""

    type: str = schema_utils.ProtectedString(AUDIO)
