from marshmallow_dataclass import dataclass

from ludwig.constants import MEAN_SQUARED_ERROR, VECTOR
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


@input_config_registry.register(VECTOR)
@dataclass
class VectorInputFeatureConfig(BaseInputFeatureConfig):
    """VectorInputFeatureConfig is a dataclass that configures the parameters used for a vector input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=VECTOR)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=VECTOR,
        default="dense",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )


@output_config_registry.register(VECTOR)
@dataclass
class VectorOutputFeatureConfig(BaseOutputFeatureConfig):
    """VectorOutputFeatureConfig is a dataclass that configures the parameters used for a vector output feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="vector_output")

    reduce_input: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce the dependencies of the output feature.",
    )

    vector_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The size of the vector. If None, the vector size will be inferred from the data.",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=VECTOR,
        default=MEAN_SQUARED_ERROR,
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=VECTOR,
        default="projector",
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    softmax: bool = schema_utils.Boolean(
        default=False,
        description="Determines whether to apply a softmax at the end of the decoder. This is useful for predicting a "
        "vector of values that sum up to 1 and can be interpreted as probabilities.",
    )
