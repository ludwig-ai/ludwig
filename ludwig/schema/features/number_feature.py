from typing import List, Tuple, Union

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    LOSS,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    MODEL_ECD,
    MODEL_GBM,
    NUMBER,
    R2,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
)
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
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
from ludwig.schema.utils import BaseMarshmallowConfig


@DeveloperAPI
@input_mixin_registry.register(NUMBER)
@dataclass
class NumberInputFeatureConfigMixin(BaseMarshmallowConfig):
    """NumberInputFeatureConfigMixin is a dataclass that configures the parameters used in both the number input
    feature and the number global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=NUMBER)


@DeveloperAPI
@dataclass
class NumberInputFeatureConfig(BaseInputFeatureConfig, NumberInputFeatureConfigMixin):
    """NumberInputFeatureConfig is a dataclass that configures the parameters used for a number input feature."""

    encoder: BaseEncoderConfig = None


@DeveloperAPI
@ecd_input_config_registry.register(NUMBER)
@dataclass(repr=False)
class ECDNumberInputFeatureConfig(NumberInputFeatureConfig):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=NUMBER,
        default="passthrough",
    )


@DeveloperAPI
@gbm_input_config_registry.register(NUMBER)
@dataclass(repr=False)
class GBMNumberInputFeatureConfig(NumberInputFeatureConfig):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_GBM,
        feature_type=NUMBER,
        default="passthrough",
    )


@DeveloperAPI
@output_mixin_registry.register(NUMBER)
@dataclass
class NumberOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """NumberOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the number output
    feature and the number global defaults section of the Ludwig Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=NUMBER,
        default="regressor",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=NUMBER,
        default=MEAN_SQUARED_ERROR,
    )


@DeveloperAPI
@output_config_registry.register(NUMBER)
@dataclass(repr=False)
class NumberOutputFeatureConfig(BaseOutputFeatureConfig, NumberOutputFeatureConfigMixin):
    """NumberOutputFeatureConfig is a dataclass that configures the parameters used for a category output
    feature."""

    clip: Union[List[int], Tuple[int]] = schema_utils.FloatRangeTupleDataclassField(
        n=2,
        default=None,
        allow_none=True,
        min=0,
        max=999999999,
        description="Clip the predicted output to the specified range.",
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [MEAN_SQUARED_ERROR],
        default=MEAN_SQUARED_ERROR,
        description="Internal only use parameter: default validation metric for number output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="number_output")

    @staticmethod
    def get_output_metric_functions():
        return {
            LOSS: None,
            MEAN_SQUARED_ERROR: None,
            MEAN_ABSOLUTE_ERROR: None,
            ROOT_MEAN_SQUARED_ERROR: None,
            ROOT_MEAN_SQUARED_PERCENTAGE_ERROR: None,
            R2: None,
        }


@DeveloperAPI
@defaults_config_registry.register(NUMBER)
@dataclass
class NumberDefaultsConfig(NumberInputFeatureConfigMixin, NumberOutputFeatureConfigMixin):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=NUMBER,
        default="passthrough",
    )
