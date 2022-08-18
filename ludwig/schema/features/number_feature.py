from marshmallow_dataclass import dataclass
from typing import List, Tuple, Union

from ludwig.constants import MISSING_VALUE_STRATEGY_OPTIONS

from ludwig.constants import DROP_ROW, NUMBER, MEAN_SQUARED_ERROR
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig, BasePreprocessingConfig
from ludwig.schema.features.utils import register_preprocessor, PreprocessingDataclassField


@register_preprocessor(NUMBER)
@dataclass
class NumberPreprocessingConfig(BasePreprocessingConfig):
    """NumberPreprocessingConfig is a dataclass that configures the parameters used for a number input feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a number column",
    )

    fill_value: float = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: float = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    normalization: str = schema_utils.StringOptions(
        ["zscore", "minmax", "log1p"],
        default=None,
        allow_none=True,
        description="Normalization strategy to use for this number feature.",
    )


@register_preprocessor("number_output")
@dataclass
class NumberOutputPreprocessingConfig(BasePreprocessingConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a number output feature",
    )


@dataclass
class NumberInputFeatureConfig(BaseInputFeatureConfig):
    """NumberInputFeatureConfig is a dataclass that configures the parameters used for a number input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=NUMBER)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type="number",
        default="passthrough",
    )


@dataclass
class NumberOutputFeatureConfig(BaseOutputFeatureConfig):
    """NumberOutputFeatureConfig is a dataclass that configures the parameters used for a category output
    feature."""

    loss: dict = schema_utils.Dict(
        default={
            "type": MEAN_SQUARED_ERROR,
            "weight": 1,
        },
        description="A dictionary containing a loss type and its hyper-parameters.",
    )

    clip: Union[List[int], Tuple[int]] = schema_utils.FloatRangeTupleDataclassField(
        n=2,
        default=None,
        allow_none=True,
        min=0,
        max=999999999,
        description="Clip the predicted output to the specified range.",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=NUMBER,
        default="regressor",
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
