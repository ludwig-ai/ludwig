from typing import Dict

from ludwig.constants import AUDIO, BINARY, CATEGORY, IMAGE, NUMBER, TEXT
from ludwig.profiling.constants import TYPES_BOOLEAN, TYPES_FRACTIONAL, TYPES_INTEGRAL, TYPES_STRING
from ludwig.profiling.metrics import are_values_audio, are_values_images, get_distinct_values, get_pct_distinct_values
from ludwig.profiling.types import ColumnProfileSummary
from ludwig.utils import strings_utils


def get_ludwig_type_from_column_profile_summary(feature_name: str, column_profile_summary: ColumnProfileSummary) -> str:
    """Returns the Ludwig type for the given feature, derived from the whylogs column profile summary."""
    distinct_values = get_distinct_values(column_profile_summary)

    # Check for unstructured types.
    if are_values_images(distinct_values, feature_name):
        return IMAGE
    if are_values_audio(distinct_values, feature_name):
        return AUDIO

    if column_profile_summary[TYPES_BOOLEAN]:
        # True booleans.
        return BINARY
    if column_profile_summary[TYPES_FRACTIONAL]:
        # True fractionals.
        return NUMBER
    if column_profile_summary[TYPES_INTEGRAL]:
        # True integers.
        # Use CATEGORY if percentage of distinct values is sufficiently low.
        if get_pct_distinct_values(column_profile_summary) < 0.5:
            return CATEGORY
        return NUMBER

    if column_profile_summary[TYPES_STRING]:
        # Check for NUMBER, CATEGORY, BINARY.
        if len(distinct_values) == 2:
            return BINARY
        if get_pct_distinct_values(column_profile_summary) < 0.5:
            return CATEGORY
        if distinct_values and strings_utils.are_all_numbers(distinct_values):
            return NUMBER
    # Fallback to TEXT.
    return TEXT


def get_ludwig_type_map_from_column_profile_summaries(
    column_profile_summaries: Dict[str, ColumnProfileSummary]
) -> Dict[str, str]:
    """Returns a map of feature name to ludwig type."""
    ludwig_type_map = {}
    for feature_name, column_profile_summary in column_profile_summaries.items():
        ludwig_type_map[feature_name] = get_ludwig_type_from_column_profile_summary(
            feature_name, column_profile_summary
        )
    return ludwig_type_map
