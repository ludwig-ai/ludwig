from typing import Set

from ludwig.profiling.constants import CARDINALITY_EST, COUNTS_N, COUNTS_NULL, FREQUENT_ITEMS
from ludwig.profiling.types import ColumnProfileSummary
from ludwig.utils.audio_utils import is_audio_score
from ludwig.utils.image_utils import is_image_score


def get_num_nonnull_values(column_profile_summary: ColumnProfileSummary) -> int:
    """Returns the number of non-null values."""
    return column_profile_summary[COUNTS_N] - column_profile_summary[COUNTS_NULL]


def get_pct_null_values(column_profile_summary: ColumnProfileSummary) -> float:
    """Returns the percentage of null values."""
    return column_profile_summary[COUNTS_NULL] / column_profile_summary[COUNTS_N]


def get_num_distinct_values(column_profile_summary: ColumnProfileSummary) -> int:
    """Returns the number of distinct values."""
    return int(column_profile_summary[CARDINALITY_EST])


def get_distinct_values(column_profile_summary: ColumnProfileSummary) -> Set[str]:
    """Returns a list of distinct values."""
    if FREQUENT_ITEMS not in column_profile_summary:
        return {}
    frequent_items = column_profile_summary[FREQUENT_ITEMS]
    if not frequent_items:
        # Can be an empty list if the feature is non-string.
        return {}
    return {frequent_item.value for frequent_item in frequent_items}


def get_pct_distinct_values(column_profile_summary: ColumnProfileSummary) -> float:
    """Returns the percentage of distinct values."""
    return get_num_distinct_values(column_profile_summary) / column_profile_summary[COUNTS_N]


def get_distinct_values_balance(column_profile_summary: ColumnProfileSummary) -> float:
    """Returns the ratio of the least frequent / the most frequent item.

    The lower the value, the more imbalanced this feature is.
    """
    if FREQUENT_ITEMS not in column_profile_summary:
        return -1
    frequent_items = column_profile_summary[FREQUENT_ITEMS]
    if not frequent_items:
        # Can be an empty list if the feature is non-string.
        return -1

    max_occurence = frequent_items[0].est
    min_occurence = frequent_items[-1].est
    return min_occurence / max_occurence


def are_values_images(distinct_values: Set[str], feature_name: str) -> bool:
    """Returns whether the given values are probably images."""
    overall_image_score = 0
    for value in distinct_values:
        overall_image_score += is_image_score(None, value, column=feature_name)
        if overall_image_score > 3:
            return True

    if overall_image_score > 0.5 * len(distinct_values):
        return True
    return False


def are_values_audio(distinct_values: Set[str], feature_name: str) -> bool:
    """Returns whether the given values are probably audio."""
    overall_audio_score = 0
    for value in distinct_values:
        overall_audio_score += is_audio_score(value)
        if overall_audio_score > 3:
            return True

    if overall_audio_score > 0.5 * len(distinct_values):
        return True
    return False
