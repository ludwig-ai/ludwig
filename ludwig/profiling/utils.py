from typing import Any, Dict, Set

from ludwig.constants import AUDIO, BINARY, CATEGORY, IMAGE, NUMBER, TEXT
from ludwig.utils import strings_utils
from ludwig.utils.audio_utils import is_audio_score
from ludwig.utils.image_utils import is_image_score


def get_num_distinct_values(column_profile_summary) -> int:
    return int(column_profile_summary["cardinality/est"])


def get_num_nonnull_values(column_profile_summary) -> int:
    return column_profile_summary["counts/n"] - column_profile_summary["counts/null"]


def get_pct_null_values(column_profile_summary) -> float:
    return column_profile_summary["counts/null"] / column_profile_summary["counts/n"]


def get_distinct_values(column_profile_summary) -> Set[str]:
    if "frequent_items/frequent_strings" not in column_profile_summary:
        return {}
    frequent_items = column_profile_summary["frequent_items/frequent_strings"]
    if not frequent_items:
        # Can be an empty list if the feature is non-string.
        return {}
    return {frequent_item.value for frequent_item in frequent_items}


def get_pct_distinct_values(column_profile_summary) -> float:
    return get_num_distinct_values(column_profile_summary) / column_profile_summary["counts/n"]


def get_distinct_values_balance(column_profile_summary) -> float:
    if "frequent_items/frequent_strings" not in column_profile_summary:
        return -1
    frequent_items = column_profile_summary["frequent_items/frequent_strings"]
    if not frequent_items:
        # Can be an empty list if the feature is non-string.
        return -1

    max_occurence = frequent_items[0].est
    min_occurence = frequent_items[-1].est
    return min_occurence / max_occurence


def are_values_images(distinct_values: Set[str], feature_name: str):
    overall_image_score = 0
    for value in distinct_values:
        is_image_score(None, value, column=feature_name)
        if overall_image_score > 3:
            return True

    if overall_image_score > 0.5 * len(distinct_values):
        return True
    return False


def are_values_audio(distinct_values: Set[str], feature_name: str):
    overall_audio_score = 0
    for value in distinct_values:
        is_audio_score(value)
        if overall_audio_score > 3:
            return True

    if overall_audio_score > 0.5 * len(distinct_values):
        return True
    return False


def get_ludwig_type_from_column_profile_summary(feature_name: str, column_profile_summary: Dict[str, Any]) -> str:
    distinct_values = get_distinct_values(column_profile_summary)

    # Check for unstructured types.
    if are_values_images(distinct_values, feature_name):
        return IMAGE
    if are_values_audio(distinct_values, feature_name):
        return AUDIO

    if column_profile_summary["types/boolean"]:
        # True booleans.
        return BINARY
    if column_profile_summary["types/fractional"]:
        # True fractionals.
        return NUMBER
    if column_profile_summary["types/integral"]:
        # True integers.
        # Use CATEGORY if percentage of distinct values is sufficiently low.
        if get_pct_distinct_values(column_profile_summary) < 0.5:
            return CATEGORY
        return NUMBER
    if column_profile_summary["types/string"]:
        # TODO: Check for DATE.
        # Check for NUMBER, CATEGORY, BINARY.
        if get_num_distinct_values(column_profile_summary) == 2:
            return BINARY
        if get_pct_distinct_values(column_profile_summary) < 0.5:
            return CATEGORY
        if strings_utils.are_all_numbers(distinct_values):
            return NUMBER
    # Fallback to TEXT.
    return TEXT
