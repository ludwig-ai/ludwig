from typing import List

from marshmallow_dataclass import dataclass

from ludwig.constants import MEAN_SQUARED_ERROR, VECTOR
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


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


@dataclass
class VectorOutputFeatureConfig(BaseOutputFeatureConfig):
    """VectorOutputFeatureConfig is a dataclass that configures the parameters used for a vector output feature."""

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

    loss: dict = schema_utils.Dict(
        default={
            "type": MEAN_SQUARED_ERROR,
            "weight": 1,
        },
        description="A dictionary containing a loss type and its hyper-parameters.",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=VECTOR,
        default="projector",
    )

    dependencies: List = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )
