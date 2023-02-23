from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import DROP_ROW, FILL_WITH_CONST, MISSING_VALUE_STRATEGY_OPTIONS, PREPROCESSING, SEQUENCE
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata import FEATURE_METADATA, PREPROCESSING_METADATA
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils import strings_utils


@DeveloperAPI
@register_preprocessor(SEQUENCE)
@ludwig_dataclass
class SequencePreprocessingConfig(BasePreprocessingConfig):
    tokenizer: str = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["tokenizer"],
    )

    vocab_file: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Filepath string to a UTF-8 encoded file containing the sequence's vocabulary. On each line the "
        "first string until \t or \n is considered a word.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["vocab_file"],
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="The maximum length (number of tokens) of the text. Texts that are longer than this value will be "
        "truncated, while texts that are shorter will be padded.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["max_sequence_length"],
    )

    most_common: int = schema_utils.PositiveInteger(
        default=20000,
        allow_none=False,
        description="The maximum number of most common tokens in the vocabulary. If the data contains more than this "
        "amount, the most infrequent symbols will be treated as unknown.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["most_common"],
    )

    padding_symbol: str = schema_utils.String(
        default=strings_utils.PADDING_SYMBOL,
        allow_none=False,
        description="The string used as a padding symbol. This special token is mapped to the integer ID 0 in the "
        "vocabulary.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["padding_symbol"],
    )

    unknown_symbol: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The string used as an unknown placeholder. This special token is mapped to the integer ID 1 in "
        "the vocabulary.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["unknown_symbol"],
    )

    padding: str = schema_utils.StringOptions(
        ["left", "right"],
        default="right",
        allow_none=False,
        description="The direction of the padding.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["padding"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["lowercase"],
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=FILL_WITH_CONST,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a text column",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["computed_fill_value"],
    )

    ngram_size: int = schema_utils.PositiveInteger(
        default=2,
        allow_none=False,
        description="The size of the ngram when using the `ngram` tokenizer (e.g, 2 = bigram, 3 = trigram, etc.).",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["ngram_size"],
    )

    cache_encoder_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Compute encoder embeddings in preprocessing, speeding up training time considerably.",
        parameter_metadata=PREPROCESSING_METADATA["cache_encoder_embeddings"],
    )


@DeveloperAPI
@register_preprocessor("sequence_output")
@ludwig_dataclass
class SequenceOutputPreprocessingConfig(SequencePreprocessingConfig):
    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a sequence output feature",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["missing_value_strategy"],
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="The maximum length (number of tokens) of the text. Texts that are longer than this value will be "
        "truncated, while texts that are shorter will be padded.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["max_sequence_length"],
    )

    tokenizer: str = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["tokenizer"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["lowercase"],
    )

    most_common: int = schema_utils.PositiveInteger(
        default=20000,
        allow_none=False,
        description="The maximum number of most common tokens in the vocabulary. If the data contains more than this "
        "amount, the most infrequent symbols will be treated as unknown.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["most_common"],
    )

    ngram_size: int = schema_utils.PositiveInteger(
        default=2,
        allow_none=False,
        description="The size of the ngram when using the `ngram` tokenizer (e.g, 2 = bigram, 3 = trigram, etc.).",
        parameter_metadata=FEATURE_METADATA[SEQUENCE][PREPROCESSING]["ngram_size"],
    )
