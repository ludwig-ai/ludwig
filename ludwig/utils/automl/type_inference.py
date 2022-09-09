from typing import Set

from ludwig.constants import AUDIO, BINARY, CATEGORY, DATE, IMAGE, NUMBER, TEXT
from ludwig.utils import strings_utils
from ludwig.utils.automl.field_info import FieldInfo

# For a given feature, the highest percentage of distinct values out of the total number of rows that we might still
# assign the CATEGORY type.
CATEGORY_TYPE_DISTINCT_VALUE_PERCENTAGE_CUTOFF = 0.5


def infer_type(field: FieldInfo, missing_value_percent: float, row_count: int) -> str:
    """Perform type inference on field.

    # Inputs
    :param field: (FieldInfo) object describing field
    :param missing_value_percent: (float) percent of missing values in the column
    :param row_count: (int) total number of entries in original dataset

    # Return
    :return: (str) feature type
    """
    if field.dtype == DATE:
        return DATE

    num_distinct_values = field.num_distinct_values
    distinct_values = field.distinct_values

    if num_distinct_values <= 1:
        return CATEGORY

    if num_distinct_values == 2 and missing_value_percent == 0:
        # Check that all distinct values are conventional bools.
        if strings_utils.are_conventional_bools(distinct_values):
            return BINARY

    if field.image_values >= 3:
        return IMAGE

    if field.audio_values >= 3:
        return AUDIO

    # Use CATEGORY if:
    # - The number of distinct values is significantly less than the total number of examples.
    # - The distinct values are not all numbers.
    # - The distinct values are all numbers but comprise of a perfectly sequential list of integers that suggests the
    #   values represent categories.
    if num_distinct_values < row_count * CATEGORY_TYPE_DISTINCT_VALUE_PERCENTAGE_CUTOFF and (
        (not strings_utils.are_all_numbers(distinct_values)) or strings_utils.are_sequential_integers(distinct_values)
    ):
        return CATEGORY

    # Use NUMBER if all of the distinct values are numbers.
    if strings_utils.are_all_numbers(distinct_values):
        return NUMBER

    # TODO (ASN): add other modalities (image, etc. )
    # Fallback to TEXT.
    return TEXT


def should_exclude(idx: int, field: FieldInfo, dtype: str, row_count: int, targets: Set[str]) -> bool:
    if field.key == "PRI":
        return True

    if field.name in targets:
        return False

    if field.num_distinct_values <= 1:
        return True

    distinct_value_percent = float(field.num_distinct_values) / row_count
    if distinct_value_percent == 1.0:
        upper_name = field.name.upper()
        if (idx == 0 and dtype == NUMBER) or upper_name.endswith("ID") or upper_name.startswith("ID"):
            return True

    return False
