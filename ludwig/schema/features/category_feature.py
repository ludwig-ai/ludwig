from marshmallow_dataclass import dataclass

from ludwig.constants import MISSING_VALUE_STRATEGY_OPTIONS

from ludwig.utils import strings_utils
from ludwig.constants import CATEGORY, SOFTMAX_CROSS_ENTROPY
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig, BasePreprocessingConfig
from ludwig.schema.features.utils import register_preprocessor, PreprocessingDataclassField


@register_preprocessor(CATEGORY)
@dataclass
class CategoryPreprocessingConfig(BasePreprocessingConfig):
    """CategoryPreprocessingConfig is a dataclass that configures the parameters used for a category input
    feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a category column",
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="Whether the string has to be lowercased before being handled by the tokenizer.",
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. if the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
    )


@dataclass
class CategoryInputFeatureConfig(BaseInputFeatureConfig):
    """CategoryInputFeatureConfig is a dataclass that configures the parameters used for a category input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=CATEGORY)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=CATEGORY,
        default="dense",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class CategoryOutputFeatureConfig(BaseOutputFeatureConfig):
    """CategoryOutputFeatureConfig is a dataclass that configures the parameters used for a category output
    feature."""

    loss: dict = schema_utils.Dict(  # TODO: Schema for loss
        default={
            "type": SOFTMAX_CROSS_ENTROPY,
            "class_weights": 1,
            "robust_lambda": 0,
            "confidence_penalty": 0,
            "class_similarities_temperature": 0,
            "weight": 1,
        },
        description="A dictionary containing a loss type and its hyper-parameters.",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=CATEGORY,
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

    top_k: int = schema_utils.NonNegativeInteger(
        default=3,
        description="Determines the parameter k, the number of categories to consider when computing the top_k "
        "measure. It computes accuracy but considering as a match if the true category appears in the "
        "first k predicted categories ranked by decoder's confidence.",
    )