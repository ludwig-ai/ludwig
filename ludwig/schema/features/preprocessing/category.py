from typing import List

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY, DROP_ROW, FILL_WITH_CONST, MISSING_VALUE_STRATEGY_OPTIONS, PREPROCESSING
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata import FEATURE_METADATA, PREPROCESSING_METADATA
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils import strings_utils


@DeveloperAPI
@register_preprocessor(CATEGORY)
@ludwig_dataclass
class CategoryPreprocessingConfig(BasePreprocessingConfig):
    """CategoryPreprocessingConfig is a dataclass that configures the parameters used for a category input
    feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=FILL_WITH_CONST,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a category column",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description=(
            "The value to replace missing values with in case the `missing_value_strategy` is `fill_with_const`"
        ),
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["computed_fill_value"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="Whether the string has to be lowercased before being handled by the tokenizer.",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["lowercase"],
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. if the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["most_common"],
    )

    cache_encoder_embeddings: bool = schema_utils.Boolean(
        default=False,
        description=(
            "For fixed encoders, compute encoder embeddings in preprocessing to avoid this step at train time. "
            "Can speed up the time taken per step during training, but will invalidate the preprocessed data "
            "if the encoder type is changed. Some model types (GBM) require caching encoder embeddings "
            "to use embedding features, and those models will override this value to `true` automatically."
        ),
        parameter_metadata=PREPROCESSING_METADATA["cache_encoder_embeddings"],
    )


@DeveloperAPI
@register_preprocessor("category_output")
@ludwig_dataclass
class CategoryOutputPreprocessingConfig(CategoryPreprocessingConfig):
    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a category output feature",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["missing_value_strategy"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="Whether the string has to be lowercased before being handled by the tokenizer.",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["lowercase"],
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. if the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["most_common"],
    )


@DeveloperAPI
@register_preprocessor("category_distribution_output")
@ludwig_dataclass
class CategoryDistributionOutputPreprocessingConfig(BasePreprocessingConfig):
    def __post_init__(self):
        if self.vocab is None:
            raise ConfigValidationError("`vocab` must be specified for `category_distribution` output feature.")

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a category output feature",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["missing_value_strategy"],
    )

    vocab: List[str] = schema_utils.List(default=None)


@DeveloperAPI
@register_preprocessor("category_llm")
@ludwig_dataclass
class LLMCategoryOutputPreprocessingConfig(CategoryOutputPreprocessingConfig):
    def __post_init__(self):
        if self.vocab is None:
            raise ConfigValidationError("`vocab` must be specified for `category_llm` output feature.")
        if self.fallback_label is None:
            raise ConfigValidationError("`fallback_label` must be specified for `category_llm` output feature.")

    vocab: List[str] = schema_utils.List(
        default=None,
        allow_none=False,
        description="The list of labels that the model can predict.",
    )

    fallback_label: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The label to use when the model doesn't match any of the labels in the `labels` list.",
    )
