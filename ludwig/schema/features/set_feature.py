from marshmallow_dataclass import dataclass

from ludwig.constants import MISSING_VALUE_STRATEGY_OPTIONS

from ludwig.utils import strings_utils

from ludwig.constants import DROP_ROW, SET, SIGMOID_CROSS_ENTROPY
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig, BasePreprocessingConfig
from ludwig.schema.features.utils import register_preprocessor, PreprocessingDataclassField


@register_preprocessor(SET)
@dataclass
class SetPreprocessingConfig(BasePreprocessingConfig):

    tokenizer: str = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to transform the raw text content of the dataset column to a set of elements. The "
        "default value space splits the string on spaces. Common options include: underscore (splits on "
        "underscore), comma (splits on comma), json (decodes the string into a set or a list through a "
        "JSON parser).",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set column",
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
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
    )


@dataclass
class SetInputFeatureConfig(BaseInputFeatureConfig):
    """SetInputFeatureConfig is a dataclass that configures the parameters used for a set input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SET)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=SET,
        default="embed",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )


@register_preprocessor("set_output")
@dataclass
class SetOutputPreprocessingConfig(BasePreprocessingConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set output feature",
    )


@dataclass
class SetOutputFeatureConfig(BaseOutputFeatureConfig):
    """SetOutputFeatureConfig is a dataclass that configures the parameters used for a set output feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="set_output")

    loss: dict = schema_utils.Dict(
        default={
            "type": SIGMOID_CROSS_ENTROPY,
            "class_weights": None,
            "weight": 1,
        },
        description="A dictionary containing a loss type and its hyper-parameters.",
    )

    threshold: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description="The threshold used to convert output probabilities to predictions. Tokens with predicted"
        "probabilities greater than or equal to threshold are predicted to be in the output set (True).",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        feature_type=SET,
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
